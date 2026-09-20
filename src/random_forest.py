import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading dataset...")

df = pd.read_csv("../data/spam_dataset.csv")
df = df[["label", "message"]].dropna()

print("Total emails:", len(df))

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training emails:", len(X_train))
print("Testing emails:", len(X_test))

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000
        )
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    )
])

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed!")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("RANDOM FOREST RESULTS")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "LEGITIMATE",
            "SPAM"
        ]
    )
)

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, predictions))