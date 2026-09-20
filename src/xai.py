import joblib

# Load improved model
model = joblib.load("../models/spam_model_improved.pkl")


def analyze_email(text):

    # Get pipeline components
    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]

    # Transform email
    X = features.transform([text])

    # Prediction
    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]

    spam_probability = probabilities[1]
    legitimate_probability = probabilities[0]

    # Feature names from both TF-IDF vectorizers
    word_vectorizer = features.transformer_list[0][1]
    char_vectorizer = features.transformer_list[1][1]

    word_features = word_vectorizer.get_feature_names_out()
    char_features = char_vectorizer.get_feature_names_out()

    all_features = list(word_features) + list(char_features)

    coefficients = classifier.coef_[0]

    # Find features present in this email
    feature_indices = X.nonzero()[1]

    spam_features = []
    legitimate_features = []

    for index in feature_indices:

        tfidf_value = X[0, index]
        contribution = float(tfidf_value * coefficients[index])

        feature = all_features[index]

        if contribution > 0:
            spam_features.append(
                (feature, contribution)
            )
        elif contribution < 0:
            legitimate_features.append(
                (feature, abs(contribution))
            )

    # Sort by strongest contribution
    spam_features.sort(
        key=lambda x: x[1],
        reverse=True
    )

    legitimate_features.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Final label
    if prediction == 1:
        label = "SPAM"
    else:
        label = "LEGITIMATE"

    return {
        "label": label,
        "spam_probability": spam_probability,
        "legitimate_probability": legitimate_probability,
        "spam_features": spam_features[:10],
        "legitimate_features": legitimate_features[:10]
    }


# Test the XAI system
if __name__ == "__main__":

    email = """
    Congratulations! You have won a $1000 prize.
    Click here immediately to claim your reward.
    """

    result = analyze_email(email)

    print("\n==============================")
    print("MAILGUARD AI - XAI")
    print("==============================")

    print(
        "Prediction:",
        result["label"]
    )

    print(
        "Spam Probability:",
        round(result["spam_probability"] * 100, 2),
        "%"
    )

    print(
        "Legitimate Probability:",
        round(result["legitimate_probability"] * 100, 2),
        "%"
    )

    print("\n⚠️ Spam-driving features:")

    for feature, score in result["spam_features"]:
        print(
            f"{feature:25} +{score:.4f}"
        )

    print("\n✓ Legitimate-driving features:")

    for feature, score in result["legitimate_features"]:
        print(
            f"{feature:25} -{score:.4f}"
        )

    print("\n==============================")