import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate(models, X_test, X_test_scaled, y_test):

    results = []

    for name, (model, xt) in models.items():
        pred = model.predict(xt)
        prob = model.predict_proba(xt)[:,1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1 Score": f1_score(y_test, pred),
            "ROC AUC": roc_auc_score(y_test, prob)
        })

    return pd.DataFrame(results)
