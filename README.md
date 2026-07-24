# Coal and Coal-Rock Quality Models for Tar Yield Prediction

## Overview

This directory contains regression models that predict tar yield (`Tar`) from coal and coal-rock quality indicators. Four machine-learning approaches are compared:

- Multilayer Perceptron (MLP)
- Random Forest (RF)
- TabPFN
- XGBoost

The code covers hyperparameter search, model training, train/test evaluation, SHAP-based interpretation, and Monte Carlo stability analysis. All Python files are standalone scripts; the project does not currently provide a unified command-line entry point.

## Data Conventions

- Data source: Excel workbooks.
- Standard input features: columns `A:N`.
- Prediction target: column `Tar`.
- Some TabPFN parameter-search scripts use columns `M:S`; verify that this range matches the current dataset before running them.
- Common worksheets: `Atrain/Atest`, `Btrain/Btest`, `Ctrain/Ctest`, and `alltrain/alltest`.
- Main evaluation metrics: MSE, RMSE, MAE, R², and MAPE.

## Directory Structure and Script Functions

```text
Model/
├─ MLP/             # Multilayer perceptron regression
├─ RF model/        # Random forest regression
├─ Tabpfn/          # TabPFN regression
├─ XGBoost/         # XGBoost regression
├─ tar-rich coal.ipynb
├─ README.md
└─ README_EN.md
```

The four model directories follow a broadly similar structure:

| File pattern | Main purpose |
| --- | --- |
| `盆地A/B/C/all.py`, `MLP盆地*.py`, `RF盆地*.py` | Train a model using predefined hyperparameters, generate predictions for the training and test sets, calculate evaluation metrics, and export observed values, predicted values, and metrics to Excel. Random forest scripts also export built-in feature importance. |
| `随机搜索*.py` | Use `RandomizedSearchCV` with five-fold cross-validation to search for hyperparameters, mainly using RMSE as the scoring metric. Results for each parameter combination and the best parameters are saved to Excel. |
| `网格搜索*.py` | Perform parameter searches for TabPFN. Despite its name, `网格搜索.py` uses randomized search, while `网格搜索C.py` iterates through a parameter grid. |
| `SHAP*.py` | Retrain the corresponding model and perform SHAP analysis. Outputs include global feature importance, beeswarm plots, interaction plots, single-feature dependence plots, and interaction plots with LOWESS trend lines. Prediction and evaluation results are also exported. |
| `MonteCarlo.py`, `MLP_MonteCarlo.py` | Repeat model training 100 times to examine the distributions and stability of R² and error metrics. The XGBoost and TabPFN scripts add Gaussian noise to the training features, while the MLP script mainly changes the random seed between runs. |

Additional notes for each directory:

- `MLP/`: Contains parameter-search, training, and SHAP scripts for datasets A, B, C, and ALL, together with an MLP Monte Carlo analysis.
- `RF model/`: Contains training and SHAP scripts for A, B, C, and ALL, plus randomized searches for A, B, and C. `SHAPRF.py` is an earlier script for dataset A and is similar to `SHAPRF-A.py`.
- `Tabpfn/`: Most scripts configure the model with `device='cuda'`. Unlike the other scripts, `Tabpfn盆地A.py` reads the `备份汇总` worksheet, creates a random 80/20 train/test split, and exports the split datasets. `SHAPTab-BBB.py` is an alternative SHAP script for dataset B.
- `XGBoost/`: Contains training and SHAP scripts for A, B, C, and ALL, a Monte Carlo analysis, and one randomized-search script.

## Purpose of `tar-rich coal.ipynb`

The notebook is an integrated analysis and publication-figure workspace. It contains 34 cells: 21 code cells and 13 Markdown/image cells. Its main functions are:

1. Plotting violin and box plots for maceral composition and performing between-group statistical tests.
2. Plotting a Spearman correlation heatmap for indicators from different basins.
3. Plotting observed-versus-predicted comparisons for RF, MLP, XGBoost, and TabPFN across different basins.
4. Integrating hyperparameter search, model training, evaluation, and SHAP analysis code for all four model types.
5. Summarizing Monte Carlo results and comparing the distributions of R² and RMSE.
6. Plotting tar yield, proximate-analysis indicators, and elemental-analysis indicators for different basins using box and violin plots.

The notebook retains selected outputs, execution logs, and embedded images. It is therefore best treated as a workspace for result verification, model comparison, and publication plotting rather than as a single end-to-end automated pipeline.

## Recommended Workflow

1. Update the hard-coded Excel input paths, figure directories, and output paths in the scripts or notebook.
2. Confirm that the selected worksheets, feature columns, and `Tar` column exist.
3. Run the parameter-search scripts to determine suitable hyperparameters for each model and basin.
4. Apply the selected parameters in the corresponding training and evaluation scripts.
5. Run the SHAP scripts to generate model-interpretation results.
6. Run the Monte Carlo scripts to evaluate model stability.

## Main Dependencies

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn openpyxl tqdm statsmodels shap xgboost tabpfn jupyter
```

The TabPFN scripts use CUDA by default. Install a PyTorch/CUDA combination compatible with the local system. If no supported GPU is available, adjust the device configuration according to the installed TabPFN version.

## Important Notes Before Running

- Most scripts contain absolute Windows paths from the original development computer. These paths must be updated when the data is moved or the code is run on another computer.
- In the current code, several basic training script names do not match the test worksheet that they actually read. Do not infer the data mapping from the filename alone:

| Script | Current training worksheet → test worksheet |
| --- | --- |
| `MLP/MLP盆地A.py` | `alltrain → alltest` |
| `MLP/MLP盆地all.py` | `alltrain → Ctest` |
| `MLP/MLP盆地B .py` | `Btrain → Atest` |
| `MLP/MLP盆地C.py` | `Ctrain → Btest` |
| `RF model/RF盆地A.py` | `Atrain → Btest` |
| `RF model/RF盆地B.py` | `Btrain → Ctest` |
| `RF model/RF盆地C.py` | `Ctrain → Btest` |
| `Tabpfn/盆地B.py` | `Btrain → Ctest` |
| `Tabpfn/盆地C.py` | `Ctrain → Btest` |
| `XGBoost/盆地ALL.py` | `alltrain → Ctest` |
| `XGBoost/盆地C.py` | `Ctrain → Btest` |
| `XGBoost/随机搜索A.py` | `alltrain → alltest` (effectively the ALL dataset) |

  These settings may represent intentional cross-basin tests, or they may be filenames and worksheet references that were not synchronized after copying scripts. Confirm the intended experiment before running them. Most other A/B/C/ALL SHAP and parameter-search scripts use matching worksheet names.

- Some SHAP scripts for the ALL dataset attempt to load `minmax_scaler.pkl` to convert normalized features back to their original units. Check the location expected by each script; some scripts will fail if the file is missing.
- The plotting scripts use the `TkAgg` backend. A non-interactive backend may be required on headless servers.
- Many output filenames are fixed, so repeated runs may overwrite existing results.
