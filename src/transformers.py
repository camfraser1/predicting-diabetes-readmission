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

class GlobalImputer(BaseEstimator, TransformerMixin):
    '''
    Custom transformer for imputing missing values in the data.

    Parameters:
    num_features: List of names of numerical features.
    cat_features : List of names of categorical features.

    '''

    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features
    
    def fit(self, X, y=None):
        '''Fit the imputer on the data by calculating median values for numerical features.'''
        X=X.copy()
        # calculate and store median of each numerical column
        self.num_imputer = X[self.num_features].median()
        return self

    def transform(self, X):
        '''Imputes values on the data'''
        X=X.copy()
        
        assert isinstance(X, pd.DataFrame), 'Expected Dataframe as input'

        # impute values
        X[self.num_features] = X[self.num_features].fillna(self.num_imputer)
        X[self.cat_features] = X[self.cat_features].fillna('missing')

        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    '''
    Custom transformer for adding engineered features to the data:
    hosp_interaction: total number of hospital interactions in the past year
    emergency_ratio: the ratio of emergency visits to total hospital interactions
    visit_intensity: the ratio between number of procedures and medications and the time spent in hospital
    '''
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        '''Adds three new numerical features to the data'''
        # check if X is a DataFrame
        X = X.copy()
        assert isinstance(X, pd.DataFrame), 'Expected Dataframe as input'
        
        # add new features
        # total hospital interactions in the past year
        X['hosp_interaction'] = X['number_outpatient'] + X['number_emergency'] + X['number_inpatient']

        # we add 1 to the denominator to avoid division by zero
        X['emergency_ratio'] = X['number_emergency'] / (X['hosp_interaction'] + 1)

        # 'time_in_hospital' has no zero values and is imputed with the median, so division is safe
        X['visit_intensity'] = (X['num_procedures'] + X['num_lab_procedures'] + X['num_medications']) / (X['time_in_hospital'])

        return X

# adds new features of the type of diagnosis for each of diag_1, diag_2, diag_3
class DiagnosisCategoriser(BaseEstimator, TransformerMixin):
    '''
    Custom transformer for reducing the cardinality of columns diag_1, diag_2, diag_3.
    
    Methods:
    categorise_diag(code)
        takes an ICD-9 code and returns the 'diagnosis category' according to ICD-9 conventions as found here:
        https://www2.gov.bc.ca/gov/content/health/practitioner-professional-resources/msp/physicians/diagnostic-code-descriptions-icd-9

    transform(X)
        takes a numpy array X (passed by the ColumnTransformer) and converts it into a DataFrame with the column names, so it can apply
        categorise_diag to each column, bucketing each code into a descriptive category and reducing cardinality enormously. 


    '''
    def __init__(self):
        # we don't include targeted columns as a parameter because they will always be diag_1, diag_2, diag_3. So we hard code them. 
        self.diagnosis_cols = ['diag_1', 'diag_2', 'diag_3']
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X=X.copy()

        # reconvert to dataframe 
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.diagnosis_cols)
            
        for col in self.diagnosis_cols:
            X[col] = X[col].apply(self.categorise_diag)
            X[col] = X[col].astype('category')

        return X

    def categorise_diag(self, code):

        if pd.isnull(code):
            return 'missing'
        # some ICD-9 codes begin with E (or e)
        if code.startswith('E') or code.startswith('e'):
            return 'external'       
        # some ICD-9 codes begin with V (or v)
        if code.startswith('V') or code.startswith('v'):
            return 'testing'
        
        # else treat it as a float
        try:
            code = float(code)
        except ValueError:
            return 'missing'
        
        if code >= 0 and code < 140:
           return 'infectious'
        elif code >= 140 and code < 240:
           return 'neoplasm'
        elif code >= 240 and code < 280:
            return 'nutritional'
        elif code >= 280 and code < 290:
            return 'blood'
        elif code >= 290 and code < 320:
            return 'mental'
        elif code >= 320 and code < 390:
            return 'nervous'
        elif code >= 390 and code < 460:
            return 'circulatory'
        elif code >= 460 and code < 520:
            return 'respiratory'
        elif code >= 520 and code < 580:
            return 'digestive'
        elif code >= 580 and code < 630:
            return 'genitourinary'
        elif code >= 630 and code < 680:
            return 'pregnancy'
        elif code >= 680 and code < 710:
            return 'skin'
        elif code >= 710 and code < 740:
            return 'musculoskeletal'
        elif code >= 740 and code < 760:
            return 'congenital'
        elif code >= 760 and code < 780:
            return 'perinatal'
        elif code >= 780 and code < 800:
            return 'ill-defined'
        elif code >= 800 and code < 1000:
            return 'injury'
        else:
            return 'missing'
        
# Transformer for categorical features - giving them ordered categories if possible and dealing with missing values
class CategorySetter(BaseEstimator, TransformerMixin):
    '''
    Custom transformer for categorical features.
    For ordinal features, it imposes custom orderings to the categories so they can be ordinally encoded properly.
    For high cardinality features, it reduces to the 10 most common values, assigning 'other' to other values.
    For nominal features, it changes type of the column to categorical and adds a 'missing' category for imputed values.

    Parameters:
    feature_names: List of names of features which are being processed

    '''
    def __init__(self, feature_names):
        '''
        We pass the names of the features being processed so we know which steps to apply.
        Other feature names are initialised here to be compared to - we can check if feature_names are nominal,
        ordinal or high-cardinality.
        We also define custom orderings for ordinal categories which can then be applied.  
        '''

        self.feature_names = feature_names
        self.nom_features = ['race', 'gender', 'payer_code','medical_specialty',
                'change', 'diabetesMed', 'admission_type_id','discharge_disposition_id',
                'admission_source_id']
        
        self.high_card_features = ['admission_type_id','discharge_disposition_id',
                'admission_source_id','medical_specialty']
        
        self.medicine_features = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide',
                     'glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose',
                     'miglitol','troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin',
                     'glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']

        self.medicine_level = ['No','Down','Steady','Up']

        self.non_med_ord_levels = {
            'age': ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                    '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'],
            'max_glu_serum': ['Norm', '>200', '>300'],
            'A1Cresult': ['Norm', '>7', '>8'],
        }
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # convert from numpy array (as passed by ColumnTransformer) to a DataFrame, so that we can access the names
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # ORDINAL FEATURE PROCESSING:

        # Combine non-medicine features with medicine features
        ordered_levels = {**self.non_med_ord_levels, **{feature: self.medicine_level for feature in self.medicine_features}}
        # add a 'missing' level for missing values
        ordered_levels = {key: ['missing'] + value for key, value in ordered_levels.items()}
        for name, levels in ordered_levels.items():
            if name in X.columns:
                X[name] = X[name].astype(pd.CategoricalDtype(categories=levels, ordered=True))

        # HIGH CARDINALITY PROCESSING: 
        for col in self.high_card_features: 
            if col in X.columns: 
                # convert column to string if it's any other type
                X[col] = X[col].astype(str)
                # define top 10 most frequent categories
                top_cats = X[col].value_counts().index[:10]
                # replace other values with 'Other'
                X[col] = X[col].where(X[col].isin(top_cats), 'Other')

        # NOMINAL FEATURE PROCESSING:
        for name in self.nom_features:
            if name in X.columns:
                X[name]=X[name].astype('category')
                # add 'missing' category
                if 'missing' not in X[name].cat.categories:
                   X[name].cat.add_categories('missing')

        return X


class SMOTENCSampler(BaseEstimator):
    '''
    Custom transformer for applying oversampling using SMOTENC. 
    This is intended as the first step of a pipeline, so it is passed the dataframe and not just a numpy array like in ColumnTransformer.

    Parameters:
        cat_features_indices: List of indices of categorical features in the dataframe (required argument for SMOTENC)
        random_state: for reproducibility

    '''
    def __init__(self, cat_features_indices, random_state=None):
        self.cat_features_indices = cat_features_indices
        self.random_state = random_state

    def fit_resample(self, X, y):
        '''
        Fits SMOTENC on X and y and resamples the data
        '''
        # Store column names
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
        else:
            raise ValueError("SMOTENCSampler expects a pandas DataFrame as input.")
        
        self.smote_ = SMOTENC(
            categorical_features=self.cat_features_indices,
            random_state=self.random_state
        )
        
        X_res, y_res = self.smote_.fit_resample(X, y)
        
        # Re-wrap X_res into a DataFrame
        X_res_df = pd.DataFrame(X_res, columns=self.columns_)
        return X_res_df, y_res
    
class UnderSampler(BaseEstimator):
    '''
    Custom transformer for applying undersampling using RandomUnderSampler. 
    This is intended as the first step of a pipeline, so it is passed the dataframe and not just a numpy array like in ColumnTransformer.

    Parameters:
        random_state: for reproducibility

    '''
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit_resample(self, X, y):
        '''
        Resamples the data
        '''
        # Store column names
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
        else:
            raise ValueError("RandomUndersampler expects a pandas DataFrame as input.")
        
        self.undersample =  RandomUnderSampler(self.random_state)
        
        X_res, y_res = self.undersample.fit_resample(X, y)
        
        # Re-wrap X_res into a DataFrame
        X_res_df = pd.DataFrame(X_res, columns=self.columns_)
        return X_res_df, y_res