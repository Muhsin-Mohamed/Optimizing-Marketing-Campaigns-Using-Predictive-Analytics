from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_models(X_train_sm, y_train_sm, X_train_scaled):

    lr = LogisticRegression(max_iter=3000)
    lr.fit(X_train_scaled, y_train_sm)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train_sm, y_train_sm)

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    xgb.fit(X_train_sm, y_train_sm)

    return lr, rf, xgb
