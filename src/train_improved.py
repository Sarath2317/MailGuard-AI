import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

# Word + Character TF-IDF
features = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True
        )
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=15000,
            sublinear_tf=True
        )
    )
])

model = Pipeline([
    ("features", features),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

print("\nTraining improved MailGuard AI model...")
model.fit(X_train, y_train)

print("Training completed!")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("IMPROVED MODEL RESULTS")
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

# Test obvious examples
test_emails = [
    "Congratulations! You have won a $1000 prize. Click here immediately to claim your reward.",
    "You won a prize click here",
    "FREE MONEY CLICK NOW",
    "URGENT! Verify your account immediately.",
    "Hi John, the meeting is scheduled for tomorrow at 10 AM."
]

print("\n==============================")
print("SAMPLE EMAIL TEST")
print("==============================")

for email in test_emails:

    probability = model.predict_proba([email])[0][1]
    prediction = model.predict([email])[0]

    label = "SPAM" if prediction == 1 else "LEGITIMATE"

    print("\nEmail:", email)
    print("Spam Probability:", round(probability * 100, 2), "%")
    print("Prediction:", label)

print("\n==============================")
print("Improved model test complete")
print("==============================")

joblib.dump(
    model,
    "../models/spam_model_improved.pkl"
)

print("\nImproved model saved successfully!")
print("../models/spam_model_improved.pkl")  