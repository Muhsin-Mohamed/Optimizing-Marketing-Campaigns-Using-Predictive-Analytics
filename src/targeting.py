import pandas as pd

def top_k_analysis(models, y_test):

    for name, (model, xt) in models.items():
        prob = model.predict_proba(xt)[:,1]

        temp = pd.DataFrame({
            "actual": y_test.values,
            "prob": prob
        }).sort_values("prob", ascending=False)

        print(f"\n{name} Targeting Results:")

        for pct in [0.10, 0.20, 0.30]:
            n = int(len(temp)*pct)
            captured = temp.head(n)["actual"].sum()
            print(f"Top {int(pct*100)}% captured = {captured}")
