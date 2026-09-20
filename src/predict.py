import joblib

model = joblib.load("../models/spam_model.pkl")


def predict_email(text):

    prediction = model.predict([text])[0]

    probability = model.predict_proba([text])[0]

    spam_probability = probability[1]
    legitimate_probability = probability[0]

    if prediction == 1:
        label = "SPAM"
    else:
        label = "LEGITIMATE"

    return {
        "label": label,
        "spam_probability": spam_probability,
        "legitimate_probability": legitimate_probability
    }


if __name__ == "__main__":

    text = input("\nEnter email text:\n")

    result = predict_email(text)

    print("\nPrediction:", result["label"])
    print(
        "Spam Probability:",
        round(result["spam_probability"] * 100, 2),
        "%"
    )