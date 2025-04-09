import pandas as pd
import matplotlib.pyplot as plt
from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model.baselines import MajorityLabelVoter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from datasets import load_dataset

from labeling import lfs, ABSTAIN
from utils import load_and_sample_imdb

df = load_and_sample_imdb(n=5000)
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df)

print("\n=== LF Summary ===")
print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())

majority_model = MajorityLabelVoter()
df["weak_label"] = majority_model.predict(L_train)

def get_responsible_lf(row):
    indices = [i for i, val in enumerate(row) if val != ABSTAIN]
    if len(indices) == 1:
        return lfs[indices[0]].name
    return "conflict_with_opinion"

df["lf_used"] = [get_responsible_lf(row) for row in L_train]
df_filtered = df[df["weak_label"] != ABSTAIN]
df_filtered.to_csv("snorkel_annotated_reviews.csv", index=False)
print("Saved annotated reviews to snorkel_annotated_reviews.csv")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df_filtered["text"])
y = df_filtered["weak_label"]

logistic_clf = LogisticRegression(max_iter=2000)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

logistic_clf.fit(X, y)
rf_clf.fit(X, y)

test_df = load_dataset("imdb")["test"].shuffle(seed=42).select(range(2000)).to_pandas()
X_test = vectorizer.transform(test_df["text"])
y_test = test_df["label"]

print("\n=== Logistic Regression Evaluation ===")
y_pred_lr = logistic_clf.predict(X_test)
lr_report = classification_report(y_test, y_pred_lr, output_dict=True)
print(classification_report(y_test, y_pred_lr))

print("\n=== Random Forest Evaluation ===")
y_pred_rf = rf_clf.predict(X_test)
rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
print(classification_report(y_test, y_pred_rf))

metrics = ['precision', 'recall', 'f1-score']
classes = ['0', '1']

for metric in metrics:
    values_lr = [lr_report[c][metric] for c in classes]
    values_rf = [rf_report[c][metric] for c in classes]

    x = range(len(classes))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width/2 for i in x], values_lr, width, label='Logistic Regression')
    plt.bar([i + width/2 for i in x], values_rf, width, label='Random Forest')
    plt.xticks(ticks=x, labels=['Negative (0)', 'Positive (1)'])
    plt.ylabel(metric.title())
    plt.title(f"{metric.title()} Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric}_comparison.png")
    plt.show()
