import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

def plot_class_distribution(y):
    sns.countplot(x=y)
    plt.title("Class Distribution")
    plt.show()

def plot_model_comparison(results_df):
    results_df.set_index("Model")[["Accuracy","Recall","F1 Score","ROC AUC"]].plot(kind="bar")
    plt.title("Model Comparison")
    plt.show()

def plot_confusion(models, y_test):
    for name, (model, xt) in models.items():
        cm = confusion_matrix(y_test, model.predict(xt))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"Confusion Matrix - {name}")
        plt.show()

def plot_roc(models, y_test):
    for name, (model, xt) in models.items():
        prob = model.predict_proba(xt)[:,1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=name)

    plt.legend()
    plt.title("ROC Curve")
    plt.show()

def plot_pr(models, y_test):
    for name, (model, xt) in models.items():
        prob = model.predict_proba(xt)[:,1]
        precision, recall, _ = precision_recall_curve(y_test, prob)
        plt.plot(recall, precision, label=name)

    plt.legend()
    plt.title("Precision-Recall Curve")
    plt.show()
