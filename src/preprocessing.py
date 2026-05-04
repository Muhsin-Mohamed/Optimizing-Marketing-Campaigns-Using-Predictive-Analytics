import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path, sep=';')

def preprocess_data(df):
    df = df.copy()
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop('y', axis=1)
    y = df['y']
    return X, y

def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
