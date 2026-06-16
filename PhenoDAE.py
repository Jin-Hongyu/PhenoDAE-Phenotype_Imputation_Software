#%% [Module imports]
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd
from time import time
import json
import argparse
import sys
import numpy as np
import random

#%% [Define DAE model (core component of PhenoDAE)]
class DAE(nn.Module):
    def __init__(self,
                 indim=80,  # input data dimension
                 width=10,  # encoding dim ratio; 10=x1.0, 20=x0.5
                 n_depth=4,  # number of layers between input layer & encoding layer
                 n_multiples=0,  # repeated layers of same dim per layer
                 nonlin=lambda dim: nn.LeakyReLU(inplace=True),  # the nonlinearity
                 verbose=False):
        super().__init__()

        outdim = indim

        if verbose:
            print('WIDTH', width)
            print('DEPTH', n_depth)
            print('MULT', n_multiples)
            print('NONLIN', nonlin)
            print('In D', indim)
            print('OutD', outdim)

        spec = []
        zdim = int(indim / width)
        zlist = list(np.linspace(indim, zdim, n_depth + 1).astype(int))
        if verbose:
            print('Encoding progression:', zlist)

        for li in range(n_depth):
            dnow = zlist[li]
            dnext = zlist[li + 1]
            spec += [(dnow, dnext)]
            if li != n_depth - 1:
                for mm in range(n_multiples):
                    spec += [(dnext, dnext)]

        if verbose:
            print('Fc layers spec:', spec)

        layers = []
        for si, (d1, d2) in enumerate(spec):
            layers += [nn.Linear(d1, d2)]
            layers += [nonlin(d2)]

        for si, (d2, d1) in enumerate(spec[::-1]):
            d2 = outdim if si == len(spec) - 1 else d2
            layers += [nn.Linear(d1, d2)]
            if si != len(spec) - 1:
                layers += [nonlin(d2)]

        self.net = nn.Sequential(*layers)

        if verbose:
            print('Zdim:', zlist[-1])

    def forward(self, x):
        x = self.net(x)
        return x

#%% [Define CopymaskDataset]
class CopymaskDataset(Dataset):
    def __init__(self, data, split, copymask_amount=0.3):
        self.data = data
        self.missing = np.isnan(data)
        self.split = split
        self.copymask_amount = copymask_amount

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        datarow = self.data[idx, ].copy()
        missing_inds = self.missing[idx, ]

        # start with an empty mask (no copy mask)
        mask_inds = np.zeros(len(datarow), dtype=bool)

        # randomly copy missing pattern from another sample
        if np.random.rand() < self.copymask_amount:
            rnd_ind = np.random.randint(self.data.shape[0])
            mask_inds = self.missing[rnd_ind, ].copy()

        observed_inds = ~missing_inds
        if np.sum(mask_inds & observed_inds) == np.sum(observed_inds):
            # avoid masking all observed values; flip one back
            mask_inds[np.where(mask_inds & observed_inds)[0]] = False

        # set true missing values to zero
        datarow[missing_inds] = 0

        return datarow, missing_inds, mask_inds

#%% [Default arguments for development]
class args:
    # Default parameters used during development (overwritten by command line)
    data_file = 'datasets/mate_female_all/data_nosex.csv'
    id_name = 'f.eid'
    lr = 0.001
    batch_size = 1024
    val_split = 0.8
    device = 'cuda:1'
    epochs = 300
    momentum = 0.9
    impute_using_saved = None  # no pretrained model by default
    output = 'datasets/mate_female_all/debug.csv'
    encoding_ratio = 1
    depth = 1
    impute_data_file = None
    copymask_amount = 0.5
    dae_noise = 0.1
    dae_noise_std = 0.3
    num_torch_threads = 8
    simulate_missing = 0.01
    bootstrap = False
    seed = -1
    quality = True
    multiple = -1
    save_model_path = None

#%% [Command line argument parser]
parser = argparse.ArgumentParser(description='PhenoDAE: Denoising Autoencoder for Phenotype Imputation')
# Required arguments
parser.add_argument('data_file', type=str, help='Path to CSV file, rows = samples, columns = features')

# Main arguments
parser.add_argument('--id_name', type=str, default='ID',
                   help='Column name used as sample identifier in the CSV')
parser.add_argument('--output', type=str,
                   help='Path to save imputed data; default: imputed_{original filename} in same folder')
parser.add_argument('--save_model_path', type=str, default=None,
                   help='Path to save model weights; default: {data_dir}/{original filename}.pth')

# Training arguments
parser.add_argument('--copymask_amount', type=float, default=0.3,
                   help='Probability of applying a copy mask to a sample (recommended 10%~50%)')
# Additional noise parameters for PhenoDAE
parser.add_argument('--dae_noise', type=float, default=0.1,
                   help='Extra noise ratio / corruption probability for PhenoDAE (recommended 0.1-0.2)')
parser.add_argument('--dae_noise_std', type=float, default=0.3,
                   help='Standard deviation of the injected Gaussian noise after phenotype standardization (default: 0.3)')
parser.add_argument('--batch_size', type=int, default=2048, help='Training batch size')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (recommended 2~0.1)')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
parser.add_argument('--val_split', type=float, default=0.8,
                   help='Validation split ratio (used for convergence monitoring)')
parser.add_argument('--device', type=str, default='cpu:0',
                   help='Computation device (cpu:0 or cuda:0)')

# Model architecture arguments
parser.add_argument('--encoding_ratio', type=float, default=1,
                   help='Ratio of encoding layer dimension to input dimension (e.g., 0.5 compresses to half)')
parser.add_argument('--depth', type=int, default=1,
                   help='Number of fully connected layers from input to bottleneck')

# Functionality switches
parser.add_argument('--save_imputed', action='store_true', default=False,
                   help='Save imputed results immediately after training')
parser.add_argument('--impute_using_saved', type=str,
                   help='Use a pretrained model (.pth) for imputation, skipping training')
parser.add_argument('--impute_data_file', type=str,
                   help='Data file to impute (different from training data)')
parser.add_argument('--seed', type=int, default=-1,
                   help='Fix random seed for reproducibility')
parser.add_argument('--bootstrap', action='store_true', default=False,
                   help='Perform bootstrap resampling on the dataset')
parser.add_argument('--multiple', type=int, default=-1,
                   help='Generate multiple imputation scripts (specify number of imputations)')
parser.add_argument('--quality', action='store_true', default=False,
                   help='Compute and output imputation quality metrics')
parser.add_argument('--simulate_missing', type=float, default=0.01,
                   help='Proportion of missing values to simulate for quality evaluation')
parser.add_argument('--num_torch_threads', type=int, default=8,
                   help='Limit number of PyTorch threads')

# Parse command line arguments (override class defaults)
args = parser.parse_args()

# Validate PhenoDAE noise hyperparameters
if args.dae_noise < 0 or args.dae_noise > 1:
    raise ValueError('--dae_noise must be between 0 and 1, representing the probability of corrupting an eligible observed entry.')
if args.dae_noise_std < 0:
    raise ValueError('--dae_noise_std must be non-negative.')

print(f'PhenoDAE noise settings: dae_noise={args.dae_noise}, dae_noise_std={args.dae_noise_std}')

#%% [Multiple imputation handling]
if args.multiple != -1:
    print('Generating multiple imputation command scripts...')
    configs = sys.argv[1:]
    mi = configs.index('--multiple')
    configs.pop(mi)
    configs.pop(mi)

    with open('multiple_imputation.sh', 'w') as fl:
        commands = [
            'python fit.py ' + ' '.join(configs) + f' --seed {m} --bootstrap --save_imputed'
            for m in range(args.multiple)
        ]
        fl.write('\n'.join(commands))
    exit()

#%% [File path configuration]
fparts = args.data_file.split('/')
save_folder = '/'.join(fparts[:-1]) + '/'
filename = args.data_file.split('/')[-1].replace('.csv', '')

save_model_path = save_folder + filename

if args.output:
    save_table_name = args.output
else:
    save_table_name = save_folder + f'imputed_{filename}'

if args.seed != -1:
    save_table_name += f'_seed{args.seed}'
    save_model_path += f'_seed{args.seed}'
if args.bootstrap:
    save_table_name += f'_bootstrap'
    save_model_path += f'_bootstrap'

save_model_path += '.pth'
if not args.output:
    save_table_name += '.csv'

if args.save_model_path is not None:
    save_model_path = args.save_model_path

if not args.impute_using_saved:
    print('Model will be saved to:', save_model_path)
if args.impute_using_saved or args.save_imputed:
    print('Imputed data will be saved to:', save_table_name)

#%% [Environment setup]
torch.set_num_threads(args.num_torch_threads)

if args.seed != -1:
    print(f'Using fixed random seed: {args.seed}')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

#%% [Data loading]
tab = pd.read_csv(args.data_file).set_index(args.id_name)
print(f'Dataset size: {tab.shape[0]} rows x {tab.shape[1]} columns')

#%% [Bootstrap sampling]
if args.bootstrap:
    print('Bootstrap sampling enabled')
    ix = list(range(len(tab)))
    ix = np.random.choice(ix, size=len(tab), replace=True)
    tab = tab.iloc[ix]
    print('First 5 sample IDs:')
    for i in tab.index[:5]:
        print(' ', i)

#%% [Feature type detection]
ncats = tab.nunique()
binary_features = tab.columns[ncats == 2]
contin_features = tab.columns[~(ncats == 2)]
feature_ord = list(contin_features) + list(binary_features)
print(f'Loaded features: continuous={len(contin_features)}, binary={len(binary_features)}')
CONT_BINARY_SPLIT = len(contin_features)

#%% [Dataset splitting]
val_ind = int(tab.shape[0] * args.val_split)
splits = ['train', 'val', 'final']
dsets = {
    'train': tab[feature_ord].iloc[:val_ind, :],
    'val': tab[feature_ord].iloc[val_ind:, :],
    'final': tab[feature_ord]
}

#%% [Standardization statistics]
train_stats = {'mean': dsets['train'].mean().values}
train_stats['std'] = np.nanstd(dsets['train'].values - train_stats['mean'], axis=0)

#%% [Standardization]
normd_dsets = {
    split: (dsets[split].values - train_stats['mean']) / train_stats['std']
    for split in splits
}

#%% [Create data loaders]
dataloaders = {}
for split, mat in normd_dsets.items():
    dataset = CopymaskDataset(
        mat,
        split,
        copymask_amount=args.copymask_amount
    )
    dataloaders[split] = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == 'train'),
        num_workers=0
    )

#%% [Model initialization]
feature_dim = dsets['train'].shape[1]
core = DAE(
    indim=feature_dim,
    width=1/args.encoding_ratio,
    n_depth=args.depth
)
model = core.to(args.device)

# Convert the statistics to tensors on the device
mean_tensor = torch.tensor(train_stats['mean'], dtype=torch.float32, device=args.device)
std_tensor = torch.tensor(train_stats['std'], dtype=torch.float32, device=args.device)

#%% [Model training]
if not args.impute_using_saved:
    cont_crit = nn.MSELoss()
    binary_crit = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        threshold=1e-10,
        patience=20
    )

    def get_lr():
        return optimizer.param_groups[0]['lr']

    hist = {
        'train': [],
        'val': [],
        'lr': [],
        'config': {
            'dae_noise': args.dae_noise,
            'dae_noise_std': args.dae_noise_std,
            'copymask_amount': args.copymask_amount,
            'encoding_ratio': args.encoding_ratio,
            'depth': args.depth,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'momentum': args.momentum,
            'val_split': args.val_split,
            'seed': args.seed
        }
    }
    best_test_loss = None

    for ep in range(args.epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            t_ep = time()
            ep_hist = {'total': [], 'binary': []}
            dset = dataloaders[phase]

            for bi, batch in enumerate(dset):
                datarow, nan_inds, train_inds = batch
                datarow = datarow.float()


                # Convert numpy masks to torch tensors on the same device as the model
                nan_inds = nan_inds.to(args.device)
                train_inds = train_inds.to(args.device)
                # Move data to device
                datarow = datarow.to(args.device)

                masked_data = datarow.clone().detach()
                masked_data[train_inds] = 0

                # PhenoDAE modification: add extra noise only during training
                if phase == 'train' and args.dae_noise > 0:
                    # Positions that are NOT copy-masked AND NOT originally missing
                    non_zero_mask = (~train_inds) & (~nan_inds)
                    noise_mask = (torch.rand_like(masked_data) < args.dae_noise) & non_zero_mask
                    if noise_mask.any():
                        noise = torch.randn_like(masked_data) * args.dae_noise_std
                        masked_data[noise_mask] = noise[noise_mask]   # Replace with Gaussian noise
                        train_inds = train_inds | noise_mask          # Mark these positions as corrupted


                existing_inds = ~nan_inds
                score_inds = existing_inds.to(args.device)


                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    yhat = model(masked_data)

                    sind = CONT_BINARY_SPLIT

                    l_cont = torch.zeros(1, device=args.device)
                    if len(contin_features) != 0:
                        l_cont = cont_crit(
                            (yhat * score_inds)[:, :sind],
                            (datarow * score_inds)[:, :sind]
                        )

                    l_binary = torch.zeros(1, device=args.device)
                    if len(binary_features) != 0:
                        # Reversing the normalization process yields the original 0/1 values
                        original_binary = (datarow[:, sind:] * std_tensor[sind:] + mean_tensor[sind:])
                        binarized = (original_binary > 0.5).float()
                        # Apply observed mask: only compute loss on originally observed entries
                        binarized = binarized * score_inds[:, sind:]
                        l_binary = binary_crit(
                            (yhat * score_inds)[:, sind:],
                            binarized
                        )






                    loss = l_cont + l_binary

                    ep_hist['total'].append(loss.item())
                    ep_hist['binary'].append(l_binary.item())

                    if np.isnan(loss.item()):
                        print("NaN detected:",
                              yhat.isnan().sum(),
                              l_cont.item(),
                              l_binary.item())

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()

                print(f'\r[E{ep+1} {phase} {bi+1}/{len(dset)}] - '
                      f'L{np.mean(ep_hist["total"]):.4f} '
                      f'({l_cont.item():.4f} {l_binary.item():.4f}) '
                      f'{time()-t_ep:.1f}s LR:{get_lr()}   ', end='')

            print()
            hist[phase].append(ep_hist['total'])
            hist['lr'].append(get_lr())

        scheduler.step(np.mean(hist['val'][-1]))

        with open(save_model_path + '.json', 'w') as fl:
            json.dump(hist, fl)

        current_loss = np.mean(hist['val'][-1])
        if best_test_loss is None or best_test_loss > current_loss:
            best_test_loss = current_loss
            torch.save(core, save_model_path)
            print('Model saved')

        if ep > 50:
            loss_1 = current_loss
            loss_50 = np.mean(hist['val'][-50])
            if loss_1 > loss_50 * 2:
                print(f'Early stopping triggered: {loss_1} > {loss_50}×2')
                break

        if np.isnan(np.mean(hist['val'][-1])):
            print('NaN detected in training, exiting...')
            break

#%% [Model loading and preparation]
if args.impute_using_saved:
    print(f'Loading pretrained weights: {args.impute_using_saved}')
    model = torch.load(args.impute_using_saved, weights_only=False)

if (args.save_imputed or args.quality) and not args.impute_using_saved:
    print('Loading best checkpoint model')
    model = torch.load(save_model_path, weights_only=False)

#%% [Data imputation]
if args.impute_data_file or args.save_imputed or args.quality:
    model = model.to(args.device)
    model.eval()

    impute_mat = args.impute_data_file if args.impute_data_file else args.data_file
    imptab = pd.read_csv(impute_mat).set_index(args.id_name)[feature_ord]
    print(f'(Imputation) Dataset size: {imptab.shape[0]}')

    mat_imptab = (imptab.values - train_stats['mean']) / train_stats['std']

    dset = DataLoader(
        CopymaskDataset(mat_imptab, 'final'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    preds_ls = []

    if args.quality:
        sim_missing = imptab.values.copy()
        print('Initial observed values count:', (~np.isnan(sim_missing)).sum())
        target_missing_sim = (~np.isnan(sim_missing)).sum() * (1 - args.simulate_missing)

        while target_missing_sim < (~np.isnan(sim_missing)).sum():
            samplesA = np.random.choice(range(len(sim_missing)), size=len(imptab)//100)
            samplesB = np.random.choice(range(len(sim_missing)), size=len(imptab)//100)

            patch = sim_missing[samplesA]
            patch[np.isnan(sim_missing[samplesB])] = np.nan
            sim_missing[samplesA] = patch

            print(f'\r Simulating missing: {target_missing_sim} < { (~np.isnan(sim_missing)).sum()}', end='')

        sim_missing = np.isnan(sim_missing)
        print()

    for bi, batch in enumerate(dset):
        datarow, _, masked_inds = batch
        datarow = datarow.float().to(args.device)

        if args.quality:
            batch_slice = slice(bi*args.batch_size, (bi+1)*args.batch_size)
            sim_mask = sim_missing[batch_slice]
            datarow[sim_mask] = 0

        with torch.no_grad():
            yhat = model(datarow)

        sind = CONT_BINARY_SPLIT
        yhat = torch.cat([
            yhat[:, :sind],
            torch.sigmoid(yhat[:, sind:])
        ], dim=1)

        preds_ls.append(yhat.cpu().numpy())
        print(f'\rImputation progress: {bi}/{len(dset)}', end='')

    pmat = np.concatenate(preds_ls)
    pmat = pmat * train_stats['std'] + train_stats['mean']
    print()

#%% [Imputation quality evaluation]
if args.quality:
    print('='*50)
    print('Imputation quality evaluation:')
    morder = np.argsort(imptab.isna().sum() / len(imptab))

    qdf = {'feature': [], 'info': [], 'r2': [], 'quality': []}

    for pi in morder:
        feature = imptab.columns[pi]
        mfrac = imptab[feature].isna().mean()

        dxstr = '(no missing)'
        var_info = None
        simr2 = 0
        flag = 'NOM'

        if mfrac > 0:
            imp_vals = pmat[:, pi][imptab[feature].isna()]
            var_imp = imp_vals.var()
            obs_vals = imptab[feature][~imptab[feature].isna()]
            var_obs = obs_vals.var()
            var_info = var_imp / var_obs

            vsim = sim_missing[:, pi] & ~imptab[feature].isna()
            if vsim.sum() > 0:
                simr2 = np.corrcoef(pmat[:, pi][vsim], imptab[feature].values[vsim])[0, 1]**2

            Nobs = (~imptab[feature].isna()).sum()
            if not np.isnan(simr2):
                Neff = simr2 * imptab[feature].isna().sum() + Nobs
            else:
                Neff = Nobs
            eff_fold = Neff / Nobs

            if mfrac < 0.1:
                flag = 'LOM'
            else:
                if var_info >= 0.2 and simr2 < 0.2:
                    flag = 'LOR'
                elif var_info < 0.2 and simr2 >= 0.2:
                    flag = 'LOV'
                elif var_info >= 0.2 and simr2 >= 0.2:
                    flag = 'QOK'
                else:
                    flag = 'LOQ'

            dxstr = f'var ratio={var_info:.2f} r²={simr2:.2f} effective sample=x{eff_fold:.1f}'

        qdf['feature'].append(feature)
        qdf['info'].append(var_info)
        qdf['r2'].append(simr2)
        qdf['quality'].append(flag)

        print(f'{flag} missing={mfrac*100:.1f}% {dxstr} {feature}')

    print('='*50)

    qdf = pd.DataFrame(qdf)
    qual_path = save_model_path.replace('.pth', '_quality.csv')
    qdf.to_csv(qual_path, index=False)

#%% [Save imputed results]
if args.impute_data_file or args.save_imputed:
    template = imptab.copy()
    tmat = template.values

    nan_mask = np.isnan(tmat)
    tmat[nan_mask] = pmat[nan_mask]

    template[:] = tmat

    template.to_csv(save_table_name)

print('Done')
