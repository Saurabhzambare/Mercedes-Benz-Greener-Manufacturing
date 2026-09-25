# Mercedes-Benz Greener Manufacturing — Exploratory Machine Learning Analysis

## Project Overview

This repository contains an exploratory machine-learning analysis based on the [Mercedes-Benz Greener Manufacturing Kaggle competition](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing). The project investigates whether anonymized vehicle configuration features can help predict the time a vehicle spends on the test bench, with the broader aim of understanding factors associated with testing efficiency.

The notebook is an earlier learning project intended to demonstrate exploratory analysis and model-based feature-importance techniques. It is not a production prediction system or a complete competition solution.

## Problem Context

Mercedes-Benz (then Daimler) provided anonymized data describing different vehicle configurations. Each training observation has a target value representing testing time. The original competition challenged participants to model that time from the supplied configuration features and thereby explore opportunities to make vehicle testing more efficient.

See the [official Kaggle competition page](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing) for the full competition description and rules.

## Dataset

The intentionally tracked datasets are located in `data/`:

| File | Description |
| --- | --- |
| `data/train.csv` | Training observations, including the target column `y`. |
| `data/test.csv` | Corresponding test observations without `y`. |

In both files, `ID` identifies an observation. Features `X0` through `X8` include important categorical configuration variables (with `X7` absent from the supplied columns), while the numerous remaining `X`-prefixed variables are primarily anonymized binary features.

## Analysis Performed

The notebook currently performs the following exploratory work:

- loads the training and test datasets and inspects their shapes;
- explores the distribution of the target variable `y`;
- examines column data types and checks the training data for missing values;
- inspects unique values, including groups of binary features;
- analyzes categorical variables and visualizes their relationships with testing time;
- compares counts and target means across binary feature values;
- inspects the distribution of `ID` in the training and test sets;
- label-encodes categorical training features;
- trains an XGBoost regressor to visualize feature importance; and
- trains a Random Forest regressor for a second feature-importance view.

These steps are exploratory. The notebook does not currently provide a completed validation workflow, final test-set predictions, a Kaggle submission, or a reported competition score.

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## Repository Structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── mercedes_benz_greener_manufacturing.ipynb
└── docs/
    └── problem_statement.docx
```

- `data/` contains the original competition CSV files used by the notebook.
- `notebooks/` contains the exploratory analysis and model experiments.
- `docs/` contains the original project problem statement.

## Running the Project

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/Saurabhzambare/Mercedes-Benz-Greener-Manufacturing.git
   cd Mercedes-Benz-Greener-Manufacturing
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, use `.venv\Scripts\Activate.ps1` instead.

3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook from the repository root:

   ```bash
   jupyter notebook
   ```

5. Open `notebooks/mercedes_benz_greener_manufacturing.ipynb`.

The notebook was developed with older versions of several libraries. Some cells use legacy Seaborn or XGBoost APIs and may require small compatibility updates when run with current releases.

## Key Learning Areas

- exploratory data analysis on mixed categorical and binary data;
- categorical feature encoding;
- visualization of feature/target relationships;
- regression-oriented machine-learning exploration;
- interpretation of tree-based feature importance; and
- comparison of XGBoost and Random Forest importance rankings.

## Current Limitations

This repository originated as an earlier learning project and retains that exploratory scope:

- some plotting calls and XGBoost parameters use legacy APIs;
- model validation is limited, with no train/validation split or cross-validation workflow;
- the notebook does not generate final test-set predictions or a submission file;
- no validated model accuracy or competition score is reported;
- preprocessing is not organized as a reusable pipeline; and
- zero-variance removal, feature engineering, and dimensionality reduction are not implemented.

These limitations provide clear opportunities to modernize the work without overstating its present capabilities.

## Future Improvements

Potential next steps include:

- modernize deprecated plotting and XGBoost APIs;
- build a reproducible preprocessing and modeling pipeline;
- identify and remove zero-variance features;
- evaluate suitable dimensionality-reduction techniques;
- introduce a train/validation split or cross-validation;
- evaluate regression models with appropriate metrics such as R²;
- compare additional regression algorithms;
- generate test-set predictions and a correctly formatted submission; and
- improve reproducibility through documented runtime versions and controlled random seeds.

These items are proposed future work and are not claimed as existing functionality.

## Data Source and Attribution

The data was supplied by Daimler/Mercedes-Benz for the [Mercedes-Benz Greener Manufacturing competition on Kaggle](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing). This repository does not claim authorship of the dataset. Use of the data remains subject to the competition's rules and terms.

## Author and Portfolio Context

Maintained as part of **Saurabh Zambare's** data science and software-development portfolio. The repository is preserved as an honest record of exploratory machine-learning practice while its presentation and project organization are improved.
