import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model.baselines import MajorityLabelVoter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from datasets import load_dataset
from labeling import lfs, ABSTAIN
from utils import load_and_sample_imdb

df = load_and_sample_imdb(n=25000)
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

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel="linear"),
}

for name, model in classifiers.items():
    model.fit(X, y)

test_df = load_dataset("imdb")["test"].shuffle(seed=42).select(range(12000)).to_pandas()
X_test = vectorizer.transform(test_df["text"])
y_test = test_df["label"]

for name, model in classifiers.items():
    print(f"\n=== {name} Evaluation ===")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_').lower()}.png")
    plt.show()

metrics = ["precision", "recall", "f1-score"]
classes = ["0", "1"]

reports = {}
for name, model in classifiers.items():
    y_pred = model.predict(X_test)
    reports[name] = classification_report(y_test, y_pred, output_dict=True)

metric_titles = {"precision": "Macro Precision", "recall": "Macro Recall", "f1-score": "Macro F1-score"}

model_names = list(classifiers.keys())

x = np.arange(len(model_names))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))

for i, metric in enumerate(metrics):
    values = [reports[name]["macro avg"][metric] for name in model_names]
    ax.bar(x + i * width, values, width, label=metric_titles[metric])

ax.set_xlabel("Models", fontsize=12)
ax.set_ylabel("Scores", fontsize=12)
ax.set_title("Comparison of Metrics Across Models", fontsize=14)
ax.set_xticks(x + width)
ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)
ax.legend(title="Metrics", fontsize=10)
plt.tight_layout()
plt.savefig("metrics_comparison.png")
plt.show()
