# Standard library
import os
import sys
import random
from pathlib import Path

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import mutual_info_classif

# Imbalanced-learn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

# XGBoost
from xgboost import XGBClassifier

# Transformers
from transformers import GlobalImputer, FeatureEngineer, SMOTENCSampler, UnderSampler, DiagnosisCategoriser, CategorySetter

def view_categorical_variables(X):
    '''Prints out each categorical variable of the dataframe, along with its unique values'''

    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

    # These columns are stored as integer types, even though they are categorical. We add them manually. 
    cat_cols += ['admission_type_id',
                'discharge_disposition_id',
                'admission_source_id']
    
    # Print out unique values to be assessed
    for col in cat_cols:
        print(f'\n{col} -> {X[col].unique()}')

    pass



def make_mi_scores(X, y):
    '''Compute Mutual Information scores on features'''
    X = X.copy()
    # quickly encode categorical features
    # MI scores requires numeric data, so we encode categorical columns here briefly to get an idea of mutual information
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()  # take the codes and put them in the column, we can ignore the labels.
    
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
    

def plot_performance(model_scores):
    '''
    Plot the results of model evaluation using the data from model_scores.
    model_scores = nested dictionary where keys are the model names and the values are dictionaries containing F1 score and average precision score. 
    '''
    # convert to a dataframe
    model_scores_df = pd.DataFrame(model_scores).T

    fig, ax = plt.subplots(figsize=(12, 6))

    # define points for x tick placement
    x = np.arange(len(model_scores_df.index))
    width = 0.25

    # build bars for F1 and Average Precision
    bars1 = ax.bar(x, model_scores_df['f1'], width=width, label='F1 Score')
    bars2 = ax.bar(x+width, model_scores_df['average_precision'], width=width, label='Average Precision')

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width/2, model_scores_df.index)
    ax.legend()
    ax.set_ylim(0,0.5)
    plt.show()

    pass


def build_pipeline(model_type, model,
                   
                   nom_cat_features,
                   ord_cat_features,
                   final_num_features,
                   cat_features_indices,

                   include_feat_eng=True, sampling='None',
                   ):
    '''
    Function that dynamically builds pipelines, depending on the following parameters.
    It follows the structure as outlined in the main notebook and shown in the diagram. 

    model_type = 'linear' or 'tree'
    model = chosen model
    include_feat_eng = whether to include engineered features
    sampling = 'smote', 'oversample' or 'None

    nom_cat_features = list of names of nominal categorical features
    ord_cat_features = list of names of ordinal categorical features
    final_num_features = list of names of numerical features INCLUDING engineered features
    cat_features_indices = list of indices of categorical features in dataframe (required for SMOTENC)

    '''

    # decide on how nominal features will be encoded - OneHot for linear models, Ordinal for tree-based
    if model_type == 'linear':
        nominal_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    elif model_type =='tree':
        nominal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    else: 
        raise ValueError('Invalid model type')
    
    # pipeline to apply to the diagnoses columns diag_1, diag_2 and diag_3 (which are nominal)
    # DiagnosisCategoriser -> Encode depending on model_type
    diag_pipeline = Pipeline(steps=[
        ('diag_categorizer', DiagnosisCategoriser()),
        ('Encoder', nominal_encoder)
    ])

    # pipeline to apply to numerical columns: only scaling
    num_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # pipeline to apply to the other nominal categorical columns
    # CategorySetter -> Encode depending on model_type
    nominal_pipeline = Pipeline(steps=[
        ('cat', CategorySetter([feature for feature in nom_cat_features])),
        ('Encoder', nominal_encoder)
    ])

    # pipeline to apply to ordinal categorical features
    # CategorySetter -> OrdinalEncoder
    ordinal_pipeline = Pipeline(steps=[
        ('cat', CategorySetter([feature for feature in ord_cat_features if feature not in ['diag_1', 'diag_2', 'diag_3']])),
        ('Ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('diag', diag_pipeline, ['diag_1', 'diag_2', 'diag_3']),
            ('num', num_pipeline, final_num_features),
            ('nom', nominal_pipeline, [feature for feature in nom_cat_features]),
            ('ord', ordinal_pipeline, [feature for feature in ord_cat_features if feature not in ['diag_1', 'diag_2', 'diag_3']])
        ]
    )

    # now combine steps depending on the arguments:
    steps = []

    # add sampling stepIF APPLICABLE
    if sampling == 'oversample':
        steps += [
            ('Oversample', SMOTENCSampler(cat_features_indices=cat_features_indices, random_state=0))
        ]
    
    if sampling == 'undersample':
        steps += [
            ('Undersample', RandomUnderSampler(random_state=0))
        ]
        
    # add imputer

    # get list of all categorical features
    cat_features = nom_cat_features + ord_cat_features + ['diag_1', 'diag_2', 'diag_3']
    # get list of numerical features before the 3 engineered features
    num_features = final_num_features[:-3]

    steps += [
        ('GlobalImputer', GlobalImputer(num_features=num_features, cat_features=cat_features))
    ]

    # add feature engineering step IF APPLICABLE
    if include_feat_eng:
        steps += [
            ('FeatureEngineer', FeatureEngineer())
        ]

    # add preprocessing and classifier
    steps += [
        ('preprocessor', clone(preprocessor)),
        ('classifier', model)
    ]

    return Pipeline(steps)

    