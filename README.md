# PINN_PK_Prediction

Physics-Informed Neural Network (PINN) for nivolumab population pharmacokinetics using a general two-compartment model.

## Repository Structure

```
PINN\_nivolumab/
├── main.py                        # Entry point: train + evaluate pipeline
├── requirements.txt
├── src/
│   ├── config.py                  # Paths, hyperparameters, random seed
│   ├── dataset.py                 # Data loading, stratified split, Dataset, DataLoader
│   ├── model.py                   # PINN\_TwoComp\_Model (concentration network + PK networks)
│   ├── loss.py                    # Data loss + physics loss (ODE residual) + parameter regularization
│   ├── train.py                   # Training loop, early stopping, CV, artifact save/load
│   └── utils.py                   # Metrics (RMSE, R², CCC, ...), predict\_fold
└── evaluate/
    ├── gof.py                     # Goodness-of-fit and weighted residual plots
    ├── pk\_distribution.py         # PK parameter distribution
    ├── shap\_analysis.py           # SHAP covariate importance analysis
    └── permutation.py             # Permutation feature importance (ΔRMSE)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1\. Configure paths

Edit `src/config.py`:

```python
DATA\_PATH  = "path/to/your/dataset.csv"
OUTPUT\_DIR = Path("path/to/your/output\_directory")
```

### 2\. Run full pipeline (train + evaluate)

```bash
python main.py
```

### 3\. Evaluate without retraining (load saved artifacts)

Uncomment Option B in `main.py`:

```python
best\_model, scalers, split\_info, all\_metrics = load\_artifacts()
```

## ODE System (2-Compartment PK)

|Compartment|ODE|
|-|-|
|Central (A1)|dA1/dt = Rate − A1·(K10 + K12) + A2·K21|
|Peripheral (A2)|dA2/dt = A1·K12 − A2·K21|

Rate (infusion input) is explicitly included in the physics loss since it is not automatically handled by the PINN framework, unlike NONMEM.

## Data Format

Required columns: `ID`, `TIME`, `AMT`, `DV`, `BW`, `EGFR`, `SEX`, `BPS`, `RAAS`
