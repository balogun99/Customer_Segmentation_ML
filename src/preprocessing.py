import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(file_path):
    """
    Load CSV, clean data, encode categorical features, scale numeric features.
    Returns:
        X_scaled: scaled and encoded feature array
        preprocessor: fitted ColumnTransformer
        df: original dataframe
    """
    df = pd.read_csv(file_path)

    # drop duplicates & missing values
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # define columns
    numeric_features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    categorical_features = ["Gender"]

    # define preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)  # encode Male/Female
    ])

    X_scaled = preprocessor.fit_transform(df)

    return X_scaled, preprocessor, df