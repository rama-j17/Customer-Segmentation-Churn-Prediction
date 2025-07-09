import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CATEGORICAL = ['country', 'gender']
NUMERIC = [
    'credit_score', 'age', 'tenure', 'balance',
    'products_number', 'credit_card', 'active_member', 'estimated_salary'
]

def load():
    df = pd.read_csv('data/Bank Customer Churn Prediction.csv')
    X = df[CATEGORICAL + NUMERIC]
    y = df['churn']
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def make_pipeline():
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first'), CATEGORICAL),
        ('num', StandardScaler(), NUMERIC)
    ])
    return preprocessor
