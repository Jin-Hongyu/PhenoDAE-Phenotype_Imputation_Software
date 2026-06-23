# PhenoDAE-Phenotype_Imputation_Software
PhenoDAE is a noise‑robust phenotype imputation framework that learns stable latent representations from incomplete and noisy phenotypic data.

## Overview of the PhenoDAE framework for phenotype imputation

![PhenoDAE framework](imags/Overview%20of%20the%20PhenoDAE%20framework%20for%20phenotype%20imputation.jpg)

## Getting Started

PhenoDAE is built with Python 3 and PyTorch. All necessary dependencies are listed in `requirements.txt`.

### With PIP
```bash
git clone https://github.com/Jin-Hongyu/PhenoDAE-Phenotype_Imputation_Software.git
cd PhenoDAE-Phenotype_Imputation_Software
pip install -r requirements.txt
```

### With Conda
```bash
git clone https://github.com/Jin-Hongyu/PhenoDAE-Phenotype_Imputation_Software.git
cd PhenoDAE-Phenotype_Imputation_Software

conda create -n PhenoDAE python=3.9
conda activate PhenoDAE
pip install -r requirements.txt
```

## Imputation tutorial

### Data preparation
PhenoDAE expects a CSV (or TSV) file where:
- The first column contains sample identifiers (e.g., `FID` or `ID`); its name can be specified via `--id_name`.
- Binary columns are automatically detected (columns containing only two unique values, typically 0/1). During training, the console will report the detected counts of continuous and binary features.

> **Demo data**  
> A small masked example dataset (`Demo_data_shrimp_masked20%.csv`) is provided in the repository. It contains 5 continuous traits and 1 binary trait, with 20% of the values randomly masked. `Demo_data_shrimp.csv` is the original file before masking.

### Training a model
Use the following command to train a denoising autoencoder on your phenotype matrix:

```bash
python PhenoDAE.py /path/to/your_data.csv \
    --id_name ID \
    --copymask_amount 0.1 \
    --batch_size 2048 \
    --epochs 10000 \
    --lr 0.1 \
    --device cpu:0 \
    --dae_noise 0.10 \
    --dae_noise_std 0.3
```

**Key parameters:**
- **`data_file` (positional argument):** path to the CSV file containing the phenotype matrix.
- `--id_name`: name of the ID column.
- `--copymask_amount`: fraction of samples for which a copy-mask is applied (0 to 1). Helps prevent overfitting. Default: `0.3`.
- `--dae_noise`: fraction of non‑missing values to which Gaussian noise is added during training (denoising). Default: `0.1`.
- `--dae_noise_std`: standard deviation of the injected Gaussian noise (applied after standardization). Default: `0.3`.
- `--batch_size`, `--epochs`, `--lr`: training hyperparameters.
- `--device`: device to use, e.g., `cpu:0` or `cuda:0`.
- `--save_imputed`: if set, will immediately impute the training data after training and save the result.

During training, the script saves the model every time the validation loss improves. The final model is stored as `<data_file_basename>.pth` in the same directory (unless `--save_model_path` is specified).

**Example – train:**

```
python PhenoDAE.py /Demo_data_shrimp_masked20%.csv --id_name ID --copymask_amount 0.1 --batch_size 2048 --epochs 10000 --lr 0.1 --device cpu:0 --dae_noise 0.10 --dae_noise_std 0.3
```

output:
```
PhenoDAE noise settings: dae_noise=0.1, dae_noise_std=0.3
Model will be saved to: .../Demo_data_shrimp_masked20%.pth
Dataset size: 2947 rows x 6 columns
Loaded features: continuous=5, binary=1
[E1 train 2/2] - L1.5484 (0.8720 0.6587) 0.8s LR:0.1
[E1 val 1/1] - L3.7363 (3.0764 0.6599) 0.0s LR:0.1
Model saved
...
Done
```

### Imputing missing values
When imputing with a saved model, you must supply the same data file that was used for training (or a file with identical column structure and similar distributions). The actual data to be imputed can be the same file or a different one via `--impute_data_file`.

```bash
python PhenoDAE.py /path/to/original_training_data.csv \
    --id_name ID \
    --impute_using_saved /path/to/saved_model.pth \
    --impute_data_file /path/to/data_with_missing.csv \
    --device cpu:0
```

- `--impute_using_saved`: path to the `.pth` model file from training.
- `--impute_data_file`: path to the CSV file you want to impute.
- The imputed matrix is saved as `imputed_<original_filename>.csv` in the same folder as the input data file, unless `--output` is specified.

**Example – imputing the same dataset used for training:**

```bash
python PhenoDAE.py /Demo_data_shrimp_masked20%.csv --id_name ID --impute_using_saved /Demo_data_shrimp_masked20%.pth --impute_data_file /Demo_data_shrimp_masked20%.csv --device cpu:0
```

Output:
```
Imputed data will be saved to: .../imputed_Demo_data_shrimp_masked20%.csv
Dataset size: 2947 rows x 6 columns
Loaded features: continuous=5, binary=1
Loading pretrained weights: .../Demo_data_shrimp_masked20%.pth
(Imputation) Dataset size: 2947
Imputation progress: 1/2
Done
```

## Citation

If you use PhenoDAE in your research, please cite our paper (coming soon).  
For now, you can reference the software as:

```bibtex

```
