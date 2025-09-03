# Predicting Diabetes Readmission

In this project, I aim to predict **30-day hospital readmissions for diabetes patients** using the [UCI Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

The dataset contains ten years of clinical data from over 130 U.S. hospitals and integrated delivery networks, and is openly licensed for public use.

The objective of the project is to build a machine learning model that is capable of predicting whether diabetes patients will be readmitted to hospital within thirty days, which would enable healthcare providers to proactively identify high-risk patients, prioritize follow-up care, and reduce avoidable costs.

## Project Structure

- `notebook.ipynb` — Final notebook containing the end-to-end workflow
- `src/`
  - `functions.py` — Utility functions used throughout the notebook
  - `transformers.py` — Custom transformers used in the pipeline
- `images/` — Contains a schematic diagram of the pipelines

## Features 

- Build custom transformers for cardinality reduction, ordinal encoding, sampling, and feature engineering
- Construct modular, robust pipelines for easy comparison and evaluation
- Evaluate model performance using F1 score, average precision, precision, and recall
- Tune hyperparameters and optimize decision thresholds
- Explore the trade-off between interpretability and predictive performance


## Installation

Follow these steps to set up the project environment on your local machine:

1. **Clone the repository**

    ```bash
    git clone https://github.com/camfraser1/predicting-diabetes-readmission.git
    cd predicting_diabetes_readmission
    ```

2. **Create virtual environment**

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install required packages**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the analysis, open the notebook with:

```bash
jupyter notebook notebook.ipynb
```


## Data

The dataset is the [UCI Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), containing over 100,000 rows of data and over 50 features, in csv format. 

The target variable is whether a patient is readmitted within thirty days. The dataset exhibits a strong class imbalance (~11% target class), which poses challenges for model training and evaluation. 



## Disclaimer
This project is for educational and research purposes only. It should not be used as a diagnostic tool. 

## License
This project is licensed under the MIT License.