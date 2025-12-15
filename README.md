# CYP2C19 Metabolizer Prediction

A machine learning pipeline for predicting drug metabolism phenotypes from CYP2C19 genetic variants, supporting pharmacogenomic clinical decision-making.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

* Python 3.8 or higher
* pip package manager
* 8GB RAM minimum
* Command line access

### Installation

Step-by-step guide to set up the development environment:

```
$ git clone <repository-url>
$ cd cyp2c19-ml-pipeline
$ pip install -r requirements.txt
```

## Usage

Run the complete machine learning pipeline:

```
$ python cyp2c19_ml_pipeline.py
```

The pipeline will:
* Generate or load patient data
* Perform data preprocessing and imputation
* Train 4 machine learning models
* Generate performance metrics and visualizations
* Save trained models and predictions

### Output Files

The pipeline generates the following outputs:
* Model performance report (TXT)
* Confusion matrices (PNG)
* Box plot visualizations (PNG)
* Predictions CSV file
* Trained model files (PKL)

## Results

* Best Model: Gradient Boosting (XGBoost)
* Accuracy: 99.2%
* AUC: 99.7%
* F1-Score: 99.2%

## Technical Details

* Dataset: 1,201 patients (655 CAD cases, 546 controls)
* Features: 6 CYP2C19 SNPs
* Target: 5 metabolizer phenotypes
* Cross-validation: 20-fold stratified

## Additional Information

* Institution: NMC Genetics Pvt. Ltd.
* Duration: May 2023 - Aug 2023
* Application: Clinical pharmacogenomics for cardiovascular disease
