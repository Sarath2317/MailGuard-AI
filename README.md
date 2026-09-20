# MailGuard AI

### Intelligent Email Spam Detection with Explainable AI

MailGuard AI is a machine-learning-based email security application designed to identify whether an email is **legitimate or spam**.

The project combines a trained machine-learning model with an interactive **Streamlit interface** and an **Explainable AI (XAI)** component, making the prediction easier to understand instead of returning only a classification result.

---

## Problem Statement

Email users receive large numbers of unwanted and potentially harmful messages every day. Traditional spam filters may classify an email correctly but often provide little explanation about **why** a message was considered suspicious.

MailGuard AI addresses this problem by providing:

* Email spam classification
* Prediction confidence/information
* Explainable prediction insights
* A simple interactive interface
* A locally trained machine-learning model

---

## Abstract

MailGuard AI is an email spam detection system developed using machine learning and Python. The system processes email content and predicts whether the message belongs to the **LEGITIMATE** or **SPAM** category.

The application uses a trained classification model and provides an interactive Streamlit interface for analyzing email content. An Explainable AI component is included to help users understand the factors contributing to the prediction.

The model achieved **97.71% accuracy** on the test set used during development.

---

## Key Features

### 📧 Email Classification

Analyze email content and classify it as:

* **LEGITIMATE**
* **SPAM**

### 🧠 Machine Learning

MailGuard AI uses a trained machine-learning classification model to learn patterns from email data and make predictions on previously unseen messages.

### 🔍 Explainable AI

The project includes an XAI module that provides additional information behind model predictions, helping make the classification process more interpretable.

### 🖥️ Interactive Interface

The application is built with Streamlit, providing a browser-based interface for testing email content without requiring a separate frontend application.

### ⚡ Local Prediction

The trained model is stored locally and loaded by the application, allowing predictions without requiring an external AI API.

---

## Model Performance

The model was trained and evaluated using a dedicated training and testing split.

| Metric          |     Result |
| --------------- | ---------: |
| Total emails    |      3,273 |
| Training emails |      2,618 |
| Testing emails  |        655 |
| Accuracy        | **97.71%** |

The reported accuracy is based on the dataset and train/test split used during project development.

---

## Technology Stack

| Technology                | Purpose                   |
| ------------------------- | ------------------------- |
| Python                    | Core development          |
| Scikit-learn              | Machine learning          |
| Joblib / Pickle           | Model persistence         |
| Streamlit                 | Web application           |
| Explainable AI techniques | Prediction interpretation |
| Git & GitHub              | Version control           |

---

## Project Structure

```text
MailGuard-AI/
│
├── .streamlit/
│   └── config.toml
│
├── chatbot/
│   └── chatbot.py
│
├── models/
│   └── spam_model_improved.pkl
│
├── src/
│   ├── __init__.py
│   ├── convert_dataset.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── random_forest.py
│   ├── train_improved.py
│   ├── train_model.py
│   └── xai.py
│
├── app.py
├── requirements.txt
└── .gitignore
```

> The raw training dataset is intentionally excluded from the repository.

---

## How It Works

```text
                 ┌─────────────────────┐
                 │    Email Content    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Text Preprocessing  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ ML Classification   │
                 │       Model         │
                 └──────────┬──────────┘
                            │
                    ┌───────┴────────┐
                    ▼                ▼
             ┌────────────┐   ┌────────────┐
             │ Legitimate │   │    Spam    │
             └────────────┘   └────────────┘
                    │                │
                    └───────┬────────┘
                            ▼
                 ┌─────────────────────┐
                 │   XAI Explanation  │
                 └─────────────────────┘
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sarath2317/MailGuard-AI.git
cd MailGuard-AI
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv myvenv
```

Activate it:

```powershell
myvenv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run MailGuard AI

```powershell
streamlit run app.py
```

The application will open in your browser.

---

## Model

The trained model used by the application is stored in:

```text
models/spam_model_improved.pkl
```

The raw training dataset is not included in the public repository.

---

## Explainable AI

A major focus of MailGuard AI is making machine-learning predictions easier to interpret.

Instead of treating the classifier as a complete black box, the project includes an XAI implementation that can provide additional information about the prediction.

The relevant implementation is available in:

```text
src/xai.py
```

---

## Results

During development, MailGuard AI achieved:

**97.71% test accuracy**

The model was evaluated using 655 test emails from the development dataset.

The application then integrates the trained model into the Streamlit interface so users can test email content interactively.

---

## Future Scope

Possible future improvements include:

* Real-time email inbox integration
* Browser extension support
* Phishing and malicious-link detection
* URL and attachment analysis
* More advanced NLP models
* Continuous model retraining
* Larger and more diverse datasets
* Improved explanation visualizations
* Multilingual email classification
* Deployment as a cloud-based security service

---

## Limitations

Machine-learning predictions depend on the quality and diversity of the training data.

The reported **97.71% accuracy** should therefore be interpreted as the performance observed on the project's evaluation dataset, rather than a guarantee of performance on every real-world email.

---

## Hackathon Project

MailGuard AI was developed as a machine-learning project focused on applying **AI, machine learning, and Explainable AI** to a practical email-security problem.

---

## License

This project is released under the **MIT License**.

---

## Author

**Sarath**

GitHub:
https://github.com/Sarath2317

---

⭐ If you find this project useful, consider giving the repository a star.
