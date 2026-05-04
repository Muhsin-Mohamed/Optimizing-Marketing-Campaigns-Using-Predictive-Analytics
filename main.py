from src.preprocessing import load_data, preprocess, split_data, apply_smote, scale_data
from src.models import train_models
from src.evaluation import evaluate
from src.visualization import *
from src.targeting import top_k_analysis

def main():

    df = load_data("data/synthetic_bank_marketing_research_dataset.csv")

    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    X_train_scaled, X_test_scaled = scale_data(X_train_sm, X_test)

    lr, rf, xgb = train_models(X_train_sm, y_train_sm, X_train_scaled)

    models = {
        "Logistic Regression": (lr, X_test_scaled),
        "Random Forest": (rf, X_test),
        "XGBoost": (xgb, X_test)
    }

    results_df = evaluate(models, X_test, X_test_scaled, y_test)

    print(results_df)

    plot_class_distribution(y)
    plot_model_comparison(results_df)
    plot_confusion(models, y_test)
    plot_roc(models, y_test)
    plot_pr(models, y_test)

    top_k_analysis(models, y_test)

    best = results_df.sort_values("F1 Score", ascending=False).iloc[0]["Model"]
    print("\nBest Model:", best)

if __name__ == "__main__":
    main()
