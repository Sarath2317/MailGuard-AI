import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MailGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# llama-3.3-70b-versatile was shut down by Groq on Aug 16, 2026.
# Set GROQ_MODEL in .env to change the model without editing code.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# ============================================================
# HTML HELPER
# ============================================================
# Strips leading whitespace and blank lines so Streamlit's Markdown
# parser never turns indented HTML into a code block (raw tags).


def h(markup, target=None):
    cleaned = "\n".join(
        line.strip()
        for line in markup.strip().splitlines()
        if line.strip()
    )
    if target is None:
        target = st
    target.markdown(cleaned, unsafe_allow_html=True)


# ============================================================
# PORTAL CSS
# ============================================================

h("""
<style>

/* ---------- Base ---------- */

.stApp {
    background-color: #f4f7fb;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- Sidebar ---------- */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1f3a 0%, #0e2a4a 60%, #0f3557 100%);
    border-right: 1px solid #1b3a5f;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] input {
    color: #0f172a !important;
    background-color: #ffffff !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] .sb-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
}

[data-testid="stSidebar"] .sb-logo {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #14b8a6, #0d9488);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
}

[data-testid="stSidebar"] .sb-name {
    font-size: 20px;
    font-weight: 800;
    line-height: 1.1;
    color: #ffffff !important;
}

[data-testid="stSidebar"] .sb-tag {
    font-size: 11px;
    color: #94a3b8 !important;
}

[data-testid="stSidebar"] .sb-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.4px;
    color: #7d93ad !important;
    margin: 20px 0 8px 2px;
}

/* Navigation radio styled as menu items */

[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 4px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    width: 100%;
    margin: 0;
    padding: 10px 12px;
    border-radius: 10px;
    border: 1px solid transparent;
    cursor: pointer;
    transition: 0.15s ease;
}

[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.07);
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(20, 184, 166, 0.30), rgba(20, 184, 166, 0.08));
    border-color: rgba(20, 184, 166, 0.60);
}

[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 14px;
    font-weight: 600;
}

/* Session card and status pill */

[data-testid="stSidebar"] .sb-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 12px;
}

[data-testid="stSidebar"] .sb-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    margin: 5px 0;
}

[data-testid="stSidebar"] .sb-spam { color: #fca5a5 !important; }
[data-testid="stSidebar"] .sb-legit { color: #86efac !important; }

[data-testid="stSidebar"] .sb-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    font-size: 12px;
    font-weight: 600;
    margin-top: 6px;
}

[data-testid="stSidebar"] .sb-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

[data-testid="stSidebar"] .sb-on {
    background: #22c55e;
    box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.25);
}

[data-testid="stSidebar"] .sb-off {
    background: #94a3b8;
}

[data-testid="stSidebar"] .sb-foot {
    margin-top: 22px;
    font-size: 11px;
    line-height: 1.7;
    color: #7d93ad !important;
}

/* Sidebar buttons */

[data-testid="stSidebar"] div.stButton > button {
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 9px;
    box-shadow: none;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: #14b8a6;
    transform: none;
}

/* ---------- Main text ---------- */

[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
[data-testid="stMain"] label {
    color: #1e293b;
}

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 {
    color: #0b1f3a;
}

/* ---------- Header banner ---------- */

.mg-header {
    background: linear-gradient(120deg, #0b1f3a 0%, #12466e 55%, #0d9488 140%);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 26px;
    box-shadow: 0 8px 24px rgba(11, 31, 58, 0.18);
}

.mg-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.25);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.4px;
    color: #e0f2fe !important;
    margin-bottom: 12px;
}

.mg-header h1 {
    margin: 0;
    font-size: 32px;
    letter-spacing: -0.5px;
    color: #ffffff !important;
}

.mg-header p {
    margin: 8px 0 0 0;
    font-size: 15px;
    color: #cbe6f5 !important;
}

/* ---------- Titles and cards ---------- */

.section-title {
    color: #0b1f3a !important;
    font-size: 24px;
    font-weight: 800;
    padding-left: 12px;
    border-left: 4px solid #0d9488;
    margin: 6px 0 12px 0;
}

.dashboard-card {
    background-color: #ffffff;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #0d9488;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    min-height: 160px;
    transition: 0.2s ease;
}

.dashboard-card:hover {
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.09);
    transform: translateY(-2px);
}

.card-detect { border-top-color: #0d9488; }
.card-explain { border-top-color: #2563eb; }
.card-assist { border-top-color: #7c3aed; }
.card-ready { border-top-color: #0b1f3a; min-height: 0; }

.card-icon {
    width: 42px;
    height: 42px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    margin-bottom: 12px;
}

.card-detect .card-icon { background: #ccfbf1; }
.card-explain .card-icon { background: #dbeafe; }
.card-assist .card-icon { background: #ede9fe; }
.card-ready .card-icon { background: #e2e8f0; }

.dashboard-card h3 {
    color: #0b1f3a !important;
    margin: 0 0 8px 0;
    font-size: 19px;
}

.dashboard-card p {
    color: #64748b !important;
    font-size: 14px;
    line-height: 1.6;
    margin: 0;
}

.footer {
    text-align: center;
    color: #94a3b8 !important;
    font-size: 12px;
    margin-top: 40px;
    padding: 15px;
}

/* ---------- Metrics ---------- */

div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #0d9488;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}

div[data-testid="stMetricLabel"] * {
    color: #64748b !important;
}

div[data-testid="stMetricValue"] * {
    color: #0b1f3a !important;
}

/* ---------- Buttons ---------- */

div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0d9488, #0f766e);
    color: #ffffff;
    border-radius: 9px;
    border: none;
    font-weight: 600;
    box-shadow: 0 2px 6px rgba(13, 148, 136, 0.25);
    transition: 0.15s ease;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #0f766e, #115e59);
    color: #ffffff;
    transform: translateY(-1px);
}

div.stButton > button * {
    color: #ffffff !important;
}

/* ---------- Inputs ---------- */

textarea {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
}

textarea:focus {
    border-color: #0d9488 !important;
    box-shadow: 0 0 0 2px rgba(13, 148, 136, 0.20) !important;
}

/* ---------- Chat and expanders ---------- */

[data-testid="stChatMessage"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}

[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}

[data-testid="stExpander"] summary * {
    color: #1e293b !important;
}

</style>
""")


# ============================================================
# CONSTANTS
# ============================================================

NAV_OPTIONS = [
    "Dashboard",
    "Email Spam Detector",
    "MailGuard Assistant",
    "Analysis History",
    "Model Performance"
]

TEST_SPAM_EMAIL = """Congratulations! You have won a $1,000,000 prize!

Click the link below immediately to claim your reward.

You must provide your bank account details and personal
information to verify your identity.

This offer expires today.
"""

TEST_LEGITIMATE_EMAIL = """Hi Team,

This is a reminder about tomorrow's project review meeting.

Please bring your updated project documentation and
presentation slides.

The meeting will begin at 10:00 AM in the seminar hall.

Regards,
Project Coordinator
"""


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_RELATIVE_PATH = os.path.join("models", "spam_model_improved.pkl")
MODEL_PATH = os.path.join(BASE_DIR, MODEL_RELATIVE_PATH)

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = MODEL_RELATIVE_PATH


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(
        "Could not load the model file "
        "`models/spam_model_improved.pkl`. Make sure it exists next to "
        "`app.py` and that scikit-learn is installed in the same "
        f"environment.\n\nError: {error}"
    )
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "nav" not in st.session_state:
    st.session_state.nav = "Dashboard"

if "history" not in st.session_state:
    st.session_state.history = []

if "email_text" not in st.session_state:
    st.session_state.email_text = ""

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ============================================================
# CALLBACKS (run before the page reruns)
# ============================================================


def go_to(page):
    st.session_state.nav = page


def load_email(text):
    st.session_state.email_text = text
    st.session_state.latest_result = None


def ask_suggested(question):
    st.session_state.pending_question = question


# ============================================================
# EMAIL ANALYSIS
# ============================================================


def analyze_email(text):

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    spam_probability = float(probabilities[1])
    legitimate_probability = float(probabilities[0])

    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]

    X = features.transform([text])

    word_vectorizer = features.transformer_list[0][1]
    char_vectorizer = features.transformer_list[1][1]

    word_features = word_vectorizer.get_feature_names_out()
    char_features = char_vectorizer.get_feature_names_out()

    all_features = list(word_features) + list(char_features)

    coefficients = classifier.coef_[0]

    feature_indices = X.nonzero()[1]

    spam_features = []
    legitimate_features = []

    for index in feature_indices:

        tfidf_value = X[0, index]

        contribution = float(
            tfidf_value * coefficients[index]
        )

        feature = all_features[index]

        if index < len(word_features):

            if contribution > 0:
                spam_features.append(
                    (feature, contribution)
                )

            elif contribution < 0:
                legitimate_features.append(
                    (feature, abs(contribution))
                )

    spam_features.sort(
        key=lambda x: x[1],
        reverse=True
    )

    legitimate_features.sort(
        key=lambda x: x[1],
        reverse=True
    )

    if prediction == 1:
        label = "SPAM"
    else:
        label = "LEGITIMATE"

    return {
        "label": label,
        "spam_probability": spam_probability,
        "legitimate_probability": legitimate_probability,
        "spam_features": spam_features[:8],
        "legitimate_features": legitimate_features[:8]
    }


# ============================================================
# GROQ ASSISTANT (EXPLANATION ONLY)
# ============================================================


def get_api_key():
    return (
        os.getenv("GROQ_API_KEY")
        or st.session_state.get("api_key_input", "")
    )


def get_client():
    key = get_api_key()
    return Groq(api_key=key) if key else None


ASSISTANT_SYSTEM_PROMPT = """
You are MailGuard Assistant, the educational AI assistant
inside the MailGuard AI spam email detection application.

Your job is to explain MailGuard's existing machine-learning
results and teach users about the system.

IMPORTANT RULES:

1. Logistic Regression is the actual email classifier.
2. Never classify an email yourself.
3. Never override the ML classifier's result.
4. Never invent spam probabilities.
5. Never invent feature contributions.
6. Only discuss probabilities and features supplied in the
   analysis context.
7. If analysis information is not available, say so clearly.
8. Explain TF-IDF, Logistic Regression, Machine Learning,
   XAI and feature contributions clearly.
9. Keep explanations understandable for a college student.
10. Do not claim that you personally detected the email.
11. Clearly distinguish between the ML result and your
    explanation of that result.
12. Never fabricate technical details.
13. ML model prediction = the actual classification.
    Groq = explanation and education only.

MailGuard architecture:

Email
→ Text Processing
→ Word TF-IDF + Character TF-IDF
→ Logistic Regression
→ SPAM / LEGITIMATE
→ XAI Feature Contributions
→ MailGuard Interface
→ Groq Assistant for explanation only.
"""


def latest_analysis_from_history():
    if not st.session_state.history:
        return None

    latest = st.session_state.history[-1]

    return {
        "label": latest["label"],
        "spam_probability": latest["spam_probability"],
        "legitimate_probability": latest.get(
            "legitimate_probability",
            1 - latest["spam_probability"]
        ),
        "spam_features": latest.get("spam_features", []),
        "legitimate_features": latest.get("legitimate_features", [])
    }


def build_analysis_context(latest_analysis):

    if not latest_analysis:
        return (
            "No email has been analyzed yet, so no analysis "
            "is available."
        )

    return f"""
LATEST MAILGUARD ANALYSIS:

Classification:
{latest_analysis.get("label", "Unknown")}

Spam Probability:
{latest_analysis.get("spam_probability", 0) * 100:.2f}%

Legitimate Probability:
{latest_analysis.get("legitimate_probability", 0) * 100:.2f}%

Spam-supporting features:
{latest_analysis.get("spam_features", [])}

Legitimate-supporting features:
{latest_analysis.get("legitimate_features", [])}
"""


def ask_mailguard_assistant(question, latest_analysis, previous_messages):

    client = get_client()

    if client is None:
        return (
            "The MailGuard Assistant is not connected yet. Set "
            "GROQ_API_KEY in the .env file or enter your Groq API key "
            "in the sidebar."
        )

    messages = [
        {
            "role": "system",
            "content": (
                ASSISTANT_SYSTEM_PROMPT
                + "\n"
                + build_analysis_context(latest_analysis)
            )
        }
    ]

    messages.extend(previous_messages[-8:])

    messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=2000
        )

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            return (
                "The assistant returned an empty response. "
                "Please try asking again."
            )

        return answer

    except Exception as error:
        return (
            "I couldn't connect to the Groq Assistant right now.\n\n"
            f"Error: {error}"
        )


# ============================================================
# RESULT RENDERING
# ============================================================


def render_result(result):

    spam_percent = result["spam_probability"] * 100
    legitimate_percent = result["legitimate_probability"] * 100

    if result["label"] == "SPAM":
        st.error(
            f"### Classification: SPAM\n"
            f"Spam Probability: {spam_percent:.2f}%"
        )
    else:
        st.success(
            f"### Classification: LEGITIMATE\n"
            f"Legitimate Probability: {legitimate_percent:.2f}%"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Spam Probability",
            f"{spam_percent:.2f}%"
        )

    with col2:
        st.metric(
            "Legitimate Probability",
            f"{legitimate_percent:.2f}%"
        )

    st.subheader("Explainable AI Analysis")

    st.write(
        "The chart shows which words pushed the Logistic Regression "
        "model toward SPAM (right) or LEGITIMATE (left)."
    )

    spam_df = pd.DataFrame(
        result["spam_features"],
        columns=["Word Feature", "Contribution"]
    )

    legitimate_df = pd.DataFrame(
        result["legitimate_features"],
        columns=["Word Feature", "Contribution"]
    )

    if spam_df.empty and legitimate_df.empty:
        st.info(
            "No strong word-level features were found for this email."
        )

    elif plt is not None:
        try:
            chart_df = pd.concat(
                [
                    spam_df.assign(Impact=spam_df["Contribution"]),
                    legitimate_df.assign(
                        Impact=-legitimate_df["Contribution"]
                    )
                ],
                ignore_index=True
            ).sort_values("Impact", ascending=True)

            colors = [
                "#dc2626" if value > 0 else "#16a34a"
                for value in chart_df["Impact"]
            ]

            fig, ax = plt.subplots(
                figsize=(9, max(3, 0.4 * len(chart_df) + 1))
            )

            ax.barh(
                chart_df["Word Feature"],
                chart_df["Impact"],
                color=colors
            )

            ax.axvline(0, linewidth=1, color="#333333")
            ax.set_xlabel("Contribution (right = SPAM, left = LEGITIMATE)")
            ax.set_title("Word Feature Contributions")

            st.pyplot(fig)
            plt.close(fig)

        except Exception as error:
            st.warning(
                f"Explainable AI chart could not be generated: {error}"
            )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Spam-supporting features**")

        if spam_df.empty:
            st.info("No strong spam-supporting words found.")
        else:
            st.dataframe(spam_df, hide_index=True)

    with col2:
        st.markdown("**Legitimate-supporting features**")

        if legitimate_df.empty:
            st.info("No strong legitimate-supporting words found.")
        else:
            st.dataframe(legitimate_df, hide_index=True)

    st.subheader("Interpretation")

    if result["label"] == "SPAM":
        st.warning(
            "The model classified this email as SPAM. Do not click "
            "links or share personal details from it."
        )
    else:
        st.info(
            "The model classified this email as LEGITIMATE. A model "
            "result is not a guarantee, so stay cautious with "
            "unexpected links and attachments."
        )

    st.caption(
        "The classification comes from the trained Logistic Regression "
        "model. The AI assistant only explains it."
    )


# ============================================================
# SIDEBAR
# ============================================================

NAV_ICONS = {
    "Dashboard": "🏠",
    "Email Spam Detector": "🔍",
    "MailGuard Assistant": "🤖",
    "Analysis History": "🕘",
    "Model Performance": "📈"
}


def session_stats_html():
    total = len(st.session_state.history)

    spam = sum(
        1
        for item in st.session_state.history
        if item["label"] == "SPAM"
    )

    legitimate = sum(
        1
        for item in st.session_state.history
        if item["label"] == "LEGITIMATE"
    )

    return f"""
    <div class="sb-card">
        <div class="sb-row"><span>Emails analyzed</span><b>{total}</b></div>
        <div class="sb-row"><span>Spam detected</span><b class="sb-spam">{spam}</b></div>
        <div class="sb-row"><span>Legitimate</span><b class="sb-legit">{legitimate}</b></div>
    </div>
    """


with st.sidebar:

    h("""
    <div class="sb-brand">
        <div class="sb-logo">🛡️</div>
        <div>
            <div class="sb-name">MailGuard AI</div>
            <div class="sb-tag">Email security portal</div>
        </div>
    </div>
    """)

    h('<div class="sb-label">NAVIGATION</div>')

    selected_module = st.radio(
        "Portal Navigation",
        NAV_OPTIONS,
        key="nav",
        format_func=lambda option: f"{NAV_ICONS[option]}  {option}",
        label_visibility="collapsed"
    )

    h('<div class="sb-label">SESSION</div>')

    stats_placeholder = st.empty()

    if st.button("Clear History", key="btn_clear_history"):
        st.session_state.history = []
        st.session_state.latest_result = None
        st.rerun()

    if st.button("Clear Assistant Chat", key="btn_clear_chat"):
        st.session_state.chat_messages = []
        st.rerun()

    h('<div class="sb-label">ASSISTANT</div>')

    if not os.getenv("GROQ_API_KEY"):
        st.text_input(
            "Groq API Key",
            type="password",
            key="api_key_input",
            help="Or set GROQ_API_KEY in the .env file."
        )

    if get_api_key():
        h("""
        <div class="sb-pill"><span class="sb-dot sb-on"></span>Assistant connected</div>
        """)
    else:
        h("""
        <div class="sb-pill"><span class="sb-dot sb-off"></span>Assistant not configured</div>
        """)

    h("""
    <div class="sb-foot">
        Logistic Regression + XAI<br>
        ML classifies · Groq explains
    </div>
    """)


# ============================================================
# HEADER
# ============================================================

h("""
<div class="mg-header">
    <div class="mg-badge">AI-POWERED EMAIL SECURITY</div>
    <h1>🛡️ MailGuard AI</h1>
    <p>Intelligent Spam Detection, Explainable AI and Email Security Support</p>
</div>
""")


# ============================================================
# DASHBOARD
# ============================================================

if selected_module == "Dashboard":

    h('<div class="section-title">MailGuard Dashboard</div>')

    st.write(
        "Welcome to MailGuard AI. Select a service from the sidebar "
        "to continue."
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        h("""
        <div class="dashboard-card card-detect">
            <div class="card-icon">🔍</div>
            <h3>Detect</h3>
            <p>
                Analyze email content and classify messages as SPAM
                or LEGITIMATE using a trained Logistic Regression model.
            </p>
        </div>
        """)

    with col2:
        h("""
        <div class="dashboard-card card-explain">
            <div class="card-icon">📊</div>
            <h3>Explain</h3>
            <p>
                See the actual words that influenced the machine-learning
                decision through feature contribution analysis.
            </p>
        </div>
        """)

    with col3:
        h("""
        <div class="dashboard-card card-assist">
            <div class="card-icon">🤖</div>
            <h3>Assist</h3>
            <p>
                Ask the MailGuard Assistant about predictions, TF-IDF,
                Logistic Regression and XAI. It explains results and
                never changes them.
            </p>
        </div>
        """)

    st.write("")

    total_emails = len(st.session_state.history)

    spam_count = sum(
        1
        for item in st.session_state.history
        if item["label"] == "SPAM"
    )

    legitimate_count = sum(
        1
        for item in st.session_state.history
        if item["label"] == "LEGITIMATE"
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric("Emails Analyzed", total_emails)

    with metric2:
        st.metric("Spam Detected", spam_count)

    with metric3:
        st.metric("Legitimate", legitimate_count)

    with metric4:
        st.metric("Model Accuracy", "98.32%")

    st.write("")

    st.subheader("Available Services")

    dashboard_data = pd.DataFrame({
        "Service": [
            "Spam Detection",
            "Feature Extraction",
            "Explainable AI Analysis",
            "AI Assistant"
        ],
        "Purpose": [
            "Classify emails as SPAM or LEGITIMATE",
            "Convert email text into numeric features",
            "Show which words influenced the decision",
            "Explain results and teach ML concepts"
        ],
        "Technology": [
            "Logistic Regression",
            "Word TF-IDF + Character TF-IDF",
            "Feature Contributions",
            "Groq LLM (explanation only)"
        ]
    })

    st.dataframe(
        dashboard_data,
        hide_index=True
    )

    h("""
    <div class="dashboard-card card-ready">
        <div class="card-icon">✅</div>
        <h3>MailGuard is ready</h3>
        <p>
            Paste an email into the spam detector to classify and explain
            it, then ask the MailGuard Assistant about the result.
        </p>
    </div>
    """)

    st.write("")

    st.button(
        "Start an analysis →",
        key="btn_start_analysis",
        on_click=go_to,
        args=("Email Spam Detector",)
    )


# ============================================================
# EMAIL SPAM DETECTOR
# ============================================================

elif selected_module == "Email Spam Detector":

    h('<div class="section-title">🔍 Email Spam Detector</div>')

    st.write(
        "Paste an email below. MailGuard classifies it with the trained "
        "Logistic Regression model and shows the words that influenced "
        "the decision."
    )

    button1, button2, button3 = st.columns(3)

    with button1:
        st.button(
            "New Analysis",
            key="btn_new_analysis",
            on_click=load_email,
            args=("",)
        )

    with button2:
        st.button(
            "Test Spam Email",
            key="btn_test_spam",
            on_click=load_email,
            args=(TEST_SPAM_EMAIL,)
        )

    with button3:
        st.button(
            "Test Legitimate Email",
            key="btn_test_legitimate",
            on_click=load_email,
            args=(TEST_LEGITIMATE_EMAIL,)
        )

    email = st.text_area(
        "Email Content",
        height=300,
        placeholder="Paste the complete email here...",
        key="email_text"
    )

    st.write("")

    if st.button(
        "Analyze Email",
        type="primary",
        key="btn_analyze"
    ):

        if not email.strip():
            st.session_state.latest_result = None
            st.warning(
                "Please enter an email before analyzing."
            )

        else:
            try:
                with st.spinner("Analyzing email..."):
                    result = analyze_email(email)

            except Exception as error:
                st.session_state.latest_result = None
                st.error(
                    f"Could not analyze this email: {error}"
                )

            else:
                st.session_state.history.append(
                    {
                        "label": result["label"],
                        "spam_probability": result["spam_probability"],
                        "legitimate_probability":
                            result["legitimate_probability"],
                        "spam_features": result["spam_features"],
                        "legitimate_features":
                            result["legitimate_features"],
                        "text": email[:250],
                        "time": datetime.now().strftime("%H:%M:%S")
                    }
                )

                st.session_state.latest_result = result

    if st.session_state.latest_result:
        render_result(st.session_state.latest_result)


# ============================================================
# MAILGUARD ASSISTANT
# ============================================================

elif selected_module == "MailGuard Assistant":

    h('<div class="section-title">🤖 MailGuard Assistant</div>')

    st.write(
        "Ask about MailGuard, machine learning, TF-IDF, XAI, or your "
        "latest email analysis. The ML model makes the classification. "
        "The assistant only explains it."
    )

    latest_analysis = latest_analysis_from_history()

    if latest_analysis:
        st.info(
            f"Latest model result: {latest_analysis['label']} "
            f"(spam probability "
            f"{latest_analysis['spam_probability'] * 100:.2f}%)"
        )
    else:
        st.info(
            "Analyze an email first if you want the assistant to "
            "explain a specific prediction."
        )

    st.caption("Suggested questions")

    q1, q2, q3 = st.columns(3)

    with q1:
        st.button(
            "How does MailGuard work?",
            key="btn_q1",
            on_click=ask_suggested,
            args=("How does MailGuard work?",)
        )

    with q2:
        st.button(
            "What is TF-IDF?",
            key="btn_q2",
            on_click=ask_suggested,
            args=("Explain TF-IDF in simple terms.",)
        )

    with q3:
        st.button(
            "Why this classification?",
            key="btn_q3",
            on_click=ask_suggested,
            args=("Why was my latest email classified this way?",)
        )

    for message in st.session_state.chat_messages:

        avatar = (
            "🤖"
            if message["role"] == "assistant"
            else "👤"
        )

        with st.chat_message(
            message["role"],
            avatar=avatar
        ):
            st.markdown(message["content"])

    pending_question = st.session_state.pop("pending_question", None)

    user_question = st.chat_input(
        "Ask MailGuard Assistant..."
    ) or pending_question

    if user_question:

        previous_messages = list(st.session_state.chat_messages)

        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_question)

        with st.chat_message("assistant", avatar="🤖"):

            with st.spinner("Preparing response..."):
                assistant_response = ask_mailguard_assistant(
                    user_question,
                    latest_analysis,
                    previous_messages
                )

            st.markdown(assistant_response)

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": assistant_response
        })


# ============================================================
# ANALYSIS HISTORY
# ============================================================

elif selected_module == "Analysis History":

    h('<div class="section-title">🕘 Analysis History</div>')

    st.write("Emails analyzed in this session and their model results.")

    if not st.session_state.history:
        st.info("No emails have been analyzed yet.")

    else:
        for item in reversed(st.session_state.history):

            with st.expander(
                f'{item["time"]}  •  {item["label"]}'
            ):
                st.write(
                    "**Classification:**",
                    item["label"]
                )
                st.write(
                    "**Spam Probability:**",
                    f'{item["spam_probability"] * 100:.2f}%'
                )
                st.write(
                    "**Legitimate Probability:**",
                    f'{item.get("legitimate_probability", 0) * 100:.2f}%'
                )

                st.write("**Email Preview:**")
                st.code(item["text"])

                st.write("**Spam-supporting features:**")

                if item.get("spam_features"):
                    for feature, contribution in item["spam_features"]:
                        st.write(f"- {feature}: +{contribution:.4f}")
                else:
                    st.write("None")

                st.write("**Legitimate-supporting features:**")

                if item.get("legitimate_features"):
                    for feature, contribution in item["legitimate_features"]:
                        st.write(f"- {feature}: +{contribution:.4f}")
                else:
                    st.write("None")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif selected_module == "Model Performance":

    h('<div class="section-title">📈 Model Performance</div>')

    st.write(
        "MailGuard AI uses a supervised machine-learning pipeline "
        "trained on spam and legitimate emails."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", "98.32%")

    with col2:
        st.metric("Spam Precision", "92%")

    with col3:
        st.metric("Spam Recall", "96.84%")

    with col4:
        st.metric("Spam F1", "94%")

    st.divider()

    st.subheader("Model Architecture")

    st.code(
        """
Email
   ↓
Text Preprocessing
   ↓
Word TF-IDF + Character TF-IDF
   ↓
Logistic Regression
   ↓
SPAM / LEGITIMATE
   ↓
XAI Feature Contributions
   ↓
MailGuard Interface
   ↓
Groq AI Assistant
(EXPLANATION ONLY)
        """,
        language="text"
    )

    st.subheader("Technology Stack")

    stack_data = pd.DataFrame({
        "Technology": [
            "Python",
            "Scikit-learn",
            "TF-IDF",
            "Logistic Regression",
            "XAI",
            "Streamlit",
            "Groq"
        ],
        "Purpose": [
            "Core programming language",
            "Machine learning",
            "Text feature extraction",
            "Email classification",
            "Feature contribution explanations",
            "Application interface",
            "AI-powered explanation assistant"
        ]
    })

    st.dataframe(
        stack_data,
        hide_index=True
    )

    st.subheader("Test Set Results")

    data1, data2, data3 = st.columns(3)

    with data1:
        st.metric("Dataset Size", "3,273 emails")

    with data2:
        st.metric("Training Emails", "2,618")

    with data3:
        st.metric("Testing Emails", "655")

    st.subheader("Confusion Matrix")

    confusion_matrix_data = pd.DataFrame(
        {
            "Predicted Legitimate": [552, 3],
            "Predicted Spam": [8, 92]
        },
        index=["Actual Legitimate", "Actual Spam"]
    )

    st.table(confusion_matrix_data)


h(session_stats_html(), stats_placeholder)


# ============================================================
# FOOTER
# ============================================================

h("""
<div class="footer">
    MailGuard AI · Explainable Spam Detection · Logistic Regression + TF-IDF + XAI + Groq Assistant
</div>
""")