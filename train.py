import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    df = pd.read_csv('train.csv')
    
    # Drop features that generally act as identifiers
    if 'Student_ID' in df.columns:
        df = df.drop(columns=['Student_ID'])
        
    print("Preparing features and targets...")
    X = df.drop(columns=['Academic_Status'])
    y = df['Academic_Status']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Identify feature types
    text_features = ['Advisor_Notes', 'Personal_Essay']
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    for f in text_features:
        if f in categorical_features:
            categorical_features.remove(f)
            
    # Create preprocessing pipelines
    print("Building preprocessing pipeline...")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    text_transformer_notes = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='')),
        ('tfidf', TfidfVectorizer(max_features=500, stop_words=None))
    ])
    
    text_transformer_essay = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='')),
        ('tfidf', TfidfVectorizer(max_features=500, stop_words=None))
    ])

    # In Scikit-learn, TfidfVectorizer expects a 1D array of strings. We need a custom wrapper or use ColumnTransformer directly on specific column values
    # SimpleImputer returns 2D, TfidfVectorizer expects 1D. We can bypass simpleimputer for text using a custom transformer,
    # OR fillna before pipeline fit. Let's handle fillna externally to keep pipeline robust.
    # We will just fillna for text columns inside a custom function
    pass

# We handle the text pipeline cleanly by just using a ColumnTransformer that passes 1D array to Tfidf
from sklearn.base import BaseEstimator, TransformerMixin

class TextImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.fillna("").astype(str).values.ravel()

def get_pipeline(X_train):
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    text_features = ['Advisor_Notes', 'Personal_Essay']
    
    for f in text_features:
        if f in categorical_features:
            categorical_features.remove(f)
            
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    transformers = [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
    
    if 'Advisor_Notes' in text_features:
        transformers.append(('text_notes', Pipeline(steps=[('imputer', TextImputer()), ('tfidf', TfidfVectorizer(max_features=300))]), 'Advisor_Notes'))
    if 'Personal_Essay' in text_features:
        transformers.append(('text_essay', Pipeline(steps=[('imputer', TextImputer()), ('tfidf', TfidfVectorizer(max_features=300))]), 'Personal_Essay'))

    preprocessor = ColumnTransformer(transformers=transformers)
    
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    return pipeline

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv('train.csv')
    
    if 'Student_ID' in df.columns:
        df = df.drop(columns=['Student_ID'])
        
    X = df.drop(columns=['Academic_Status'])
    y = df['Academic_Status']
    
    print("Training model...")
    pipeline = get_pipeline(X)
    pipeline.fit(X, y)
    
    # Train validation
    print("Evaluating...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    val_pipeline = get_pipeline(X_train)
    val_pipeline.fit(X_train, y_train)
    y_pred = val_pipeline.predict(X_val)
    print(f"Validation F1 Macro: {f1_score(y_val, y_pred, average='macro'):.4f}")
    
    print("Saving model to model.pkl...")
    joblib.dump(pipeline, 'model.pkl')
    # Save a small sample for default values in the app
    X.median(numeric_only=True).to_pickle('defaults.pkl')
    print("Done!")
