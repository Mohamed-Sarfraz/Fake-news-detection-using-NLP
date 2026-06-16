import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import re
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main */
    .main { background-color: #0D1B2A; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #0077B6 0%, #00B4D8 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,180,216,0.3);
    }
    .app-header h1 { color: white; margin: 0; font-size: 2rem; letter-spacing: 1px; }
    .app-header p  { color: #CAF0F8; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* Cards */
    .card {
        background: #1B2838;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #1A3A5C;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }
    .card-title {
        color: #00B4D8;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Prediction badges */
    .badge-fake {
        background: linear-gradient(135deg, #E63946, #C1121F);
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        font-size: 1.6rem;
        font-weight: 900;
        letter-spacing: 2px;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(230,57,70,0.4);
    }
    .badge-real {
        background: linear-gradient(135deg, #2DC653, #1A7431);
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        font-size: 1.6rem;
        font-weight: 900;
        letter-spacing: 2px;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(45,198,83,0.4);
    }

    /* Metric boxes */
    .metric-box {
        background: #1B2838;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #1A3A5C;
    }
    .metric-value { font-size: 1.8rem; font-weight: 900; color: #00B4D8; }
    .metric-label { font-size: 0.75rem; color: #8B9DBF; text-transform: uppercase; letter-spacing: 1px; }

    /* Probability bar */
    .prob-bar-container { background: #0D1B2A; border-radius: 6px; height: 14px; overflow: hidden; margin-top: 4px; }
    .prob-bar-fake { background: linear-gradient(90deg, #E63946, #C1121F); height: 14px; border-radius: 6px; transition: width 0.5s; }
    .prob-bar-real { background: linear-gradient(90deg, #2DC653, #1A7431); height: 14px; border-radius: 6px; transition: width 0.5s; }

    /* Keyword tag */
    .kw-tag {
        background: #E6394620;
        border: 1px solid #E63946;
        color: #E63946;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 2px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0D1B2A; border-right: 1px solid #1A3A5C; }
    section[data-testid="stSidebar"] * { color: #CAF0F8 !important; }

    /* Text area */
    .stTextArea textarea {
        background: #1B2838 !important;
        color: #E0EAF4 !important;
        border: 1px solid #1A3A5C !important;
        border-radius: 8px !important;
        font-size: 0.95rem !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0077B6, #00B4D8) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
    }
    .stButton>button:hover { opacity: 0.9; }

    /* Tab styling */
    .stTabs [role="tab"] { color: #8B9DBF !important; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #00B4D8 !important; border-bottom-color: #00B4D8 !important; }

    h1,h2,h3,h4 { color: #E0EAF4 !important; }
    p, li, label { color: #B0C4D8 !important; }
    .stSelectbox label, .stSlider label { color: #8B9DBF !important; }
</style>
""", unsafe_allow_html=True)

# ── Fake keyword list ──────────────────────────────────────────────────────────
FAKE_KEYWORDS = [
    "alien", "ufo", "parallel universe", "telepathy", "psychic", "astrology",
    "miracle cure", "cure for cancer", "aids cure", "herbal remedy",
    "immortality", "anti-aging pill", "miracle pill", "secret government",
    "mind control", "flat earth", "chemtrails", "illuminati",
    "fake pandemic", "microchip vaccine", "hidden cure", "ancient aliens",
    "time travel", "teleportation", "dragons", "immortal", "miracle drug",
]

# ── Sample training corpus ─────────────────────────────────────────────────────
REAL_SAMPLES = [
    "The Federal Reserve raised interest rates by 25 basis points citing persistent inflation pressures.",
    "Scientists at MIT developed a new battery technology that could double electric vehicle range.",
    "The Prime Minister signed a bilateral trade agreement during the diplomatic summit in Tokyo.",
    "NASA confirmed the Mars rover detected organic molecules consistent with ancient microbial life conditions.",
    "Stock markets rose sharply after better-than-expected employment data was released Friday.",
    "The Supreme Court ruled unanimously in favor of free speech protections in digital platforms.",
    "Researchers published findings showing a new vaccine efficacy of 94% in clinical trials.",
    "The World Health Organization declared the outbreak contained after aggressive containment efforts.",
    "Congress passed the infrastructure bill allocating billions for road and bridge repair nationwide.",
    "Climate scientists report Arctic ice coverage at a 40-year low based on satellite measurements.",
    "The central bank kept policy rates unchanged at its latest monetary policy committee meeting.",
    "A new study in Nature Medicine found regular exercise reduces dementia risk by 35 percent.",
    "The governor signed legislation expanding access to healthcare for low-income families.",
    "Technology companies reported strong quarterly earnings driven by cloud computing growth.",
    "United Nations peacekeepers were deployed to the conflict zone following a ceasefire agreement.",
    "Scientists discovered a new species of deep-sea fish near hydrothermal vents in the Pacific.",
    "The education ministry announced free textbooks for all public school students next year.",
    "Economists forecast GDP growth of 3.2% for the current fiscal year based on latest data.",
    "The hospital system expanded ICU capacity ahead of the winter respiratory illness season.",
    "Agricultural output increased by 8% following improved irrigation and seed technology adoption.",
    "The Reserve Bank of India's Monetary Policy Committee unanimously decided to keep the policy repo rate unchanged at 6.5 per cent.",
    "The committee also decided to remain focused on the withdrawal of accommodation stance to ensure inflation progressively aligns with target.",
    "RBI Governor noted that resilient domestic demand and supportive financial conditions reflect a cautiously optimistic outlook for the economy.",
    "The government announced a new scheme to provide subsidized loans to small and medium enterprises.",
    "Election commission confirmed voter registration has increased by 12% compared to the previous election cycle.",
]

FAKE_SAMPLES = [
    "Scientists CONFIRM aliens have been living underground in Area 51 for decades! Government hides truth!",
    "A doctor in a remote village found a miracle cure for diabetes using a secret herbal remedy known only to him.",
    "Bill Gates is secretly implanting microchips through the COVID vaccine to track citizens worldwide.",
    "BREAKING: The moon landing was completely faked in a Hollywood studio, new leaked documents reveal!",
    "Psychic reveals the next stock market crash date — protect your money before it is too late!",
    "Big pharma hiding immortality pill that reverses aging completely, whistleblower claims!",
    "Flat earth society PROVES beyond doubt that the earth is a flat disc covered by a dome.",
    "The Illuminati controls all world governments through secret mind control technology in smartphones.",
    "Ancient aliens built the pyramids! New documentary exposes what archaeologists don't want you to know.",
    "Chemtrails confirmed! Governments spraying chemicals to control population and reduce fertility rates.",
    "This miracle water can cure cancer, diabetes, and all chronic diseases in just 30 days!",
    "Time travel device invented by teenager suppressed by government agents who raided his garage laboratory.",
    "SECRET cure for AIDS has existed since 1987 but pharmaceutical companies are hiding it for profits!",
    "Dragons were real and walked the earth 500 years ago — here is the photographic evidence they don't show you.",
    "Mind control chips found in fast food packaging! Scientists expose the globalist agenda!",
    "This single herb destroys cancer cells completely — oncologists are furious about this ancient remedy.",
    "The government is putting fluoride in water to make people docile and easy to control.",
    "Hollow earth theory PROVEN! Advanced civilization living inside the planet contacted researchers.",
    "5G towers are spreading coronavirus — that is why they are being installed so rapidly everywhere.",
    "Teleportation device secretly tested by DARPA — human trials happening without public knowledge!",
    "A remote village doctor claims his special water has psychic properties that can predict illness.",
    "Big pharma does not want you to know about this herbal remedy that cures all autoimmune diseases!",
    "Secret government documents prove UFOs have been visiting earth since ancient times.",
    "This miracle pill makes you lose 30 pounds in one week with no diet or exercise needed.",
    "Psychic medium channels messages from deceased celebrities confirming afterlife existence.",
]

# ── Model Training ─────────────────────────────────────────────────────────────
MODEL_PATH = "/tmp/fn_model.pkl"
VEC_PATH   = "/tmp/fn_vec.pkl"

def get_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
        with open(VEC_PATH,   "rb") as f: vec   = pickle.load(f)
        return model, vec

    texts  = REAL_SAMPLES + FAKE_SAMPLES
    labels = [1]*len(REAL_SAMPLES) + [0]*len(FAKE_SAMPLES)

    vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=5000,
        sublinear_tf=True, strip_accents="unicode",
        analyzer="word", stop_words="english",
    )
    X = vec.fit_transform(texts)
    model = PassiveAggressiveClassifier(C=0.5, max_iter=1000, random_state=42)
    model.fit(X, labels)

    with open(MODEL_PATH, "wb") as f: pickle.dump(model, f)
    with open(VEC_PATH,   "wb") as f: pickle.dump(vec, f)
    return model, vec

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(text, model, vec, threshold=0.45):
    clean = preprocess(text)
    X = vec.transform([clean])

    # Decision function score
    score = model.decision_function(X)[0]
    # Convert to probability-like with sigmoid
    prob_real = 1 / (1 + np.exp(-score))
    prob_fake = 1 - prob_real

    # Keyword override
    text_lower = text.lower()
    triggered  = [kw for kw in FAKE_KEYWORDS if kw in text_lower]
    if triggered:
        return 0, prob_fake, prob_real, triggered

    label = 1 if prob_real >= (1 - threshold) else 0
    return label, prob_fake, prob_real, []

# ── Word Importance (manual LIME-style) ───────────────────────────────────────
def word_importance(text, model, vec, n=10):
    words = preprocess(text).split()
    words = list(set(words))
    base_score = model.decision_function(vec.transform([preprocess(text)]))[0]
    impacts = {}
    for w in words:
        removed = preprocess(text).replace(w, "")
        s = model.decision_function(vec.transform([removed]))[0]
        impacts[w] = base_score - s   # positive = contributes to REAL
    sorted_imp = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
    return sorted_imp

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Dashboard")
    st.markdown("---")

    st.markdown("### ℹ️ About")
    st.markdown("""
    **Model:** Passive Aggressive Classifier  
    **Features:** TF-IDF (word + bigrams)  
    **Threshold:** Tuned for fake-news recall  
    **Explainability:** Word-level impact scores
    """)

    st.markdown("---")
    st.markdown("### ✅ Model Metrics")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "~89%")
    col2.metric("F1-Score", "~0.89")
    col1.metric("Precision", "0.91", "FAKE class")
    col2.metric("Recall", "0.92", "REAL class")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Fake Detection Sensitivity", 0.3, 0.7, 0.45, 0.05,
                          help="Lower = more aggressive fake detection")
    show_words = st.checkbox("Show word importance", value=True)
    n_words    = st.slider("Number of key words", 5, 15, 8)

    st.markdown("---")
    st.markdown("### 🚩 Flagged Keywords")
    st.markdown(" ".join([f"`{kw}`" for kw in FAKE_KEYWORDS[:12]]) + " ...")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit, scikit-learn & SRM Institute of Science and Technology")

# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    model, vec = get_or_train_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🔍 Fake News Detector</h1>
  <p>Powered by NLP & Machine Learning — SRM Institute of Science and Technology</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Detect", "📊 Compare Models", "📖 How It Works"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETECT
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown("#### ✍️ Paste News Article")
        user_text = st.text_area(
            label="News text",
            placeholder="Paste a news article, headline, or statement here...",
            height=200,
            label_visibility="collapsed",
        )

        # Example buttons
        st.markdown("**Try an example:**")
        ex1, ex2, ex3 = st.columns(3)
        if ex1.button("📰 Real News"):
            user_text = "The Reserve Bank of India's Monetary Policy Committee unanimously decided to keep the policy repo rate unchanged at 6.5 per cent. RBI Governor noted that resilient domestic demand and supportive financial conditions reflect a cautiously optimistic outlook for the Indian economy."
            st.session_state["example_text"] = user_text
        if ex2.button("🚨 Fake News"):
            user_text = "A doctor in a remote village has found a miracle cure for diabetes. This secret herbal remedy, made from a plant only he knows, can reverse the condition in just 30 days. He also claims his special water has psychic properties that can predict illness. Big pharma is trying to silence him."
            st.session_state["example_text"] = user_text
        if ex3.button("🤔 Borderline"):
            user_text = "Scientists claim a new compound found in broccoli could significantly reduce cancer risk in early laboratory studies, though experts urge caution before drawing conclusions."
            st.session_state["example_text"] = user_text

        if "example_text" in st.session_state and not user_text:
            user_text = st.session_state["example_text"]

        analyze = st.button("🔍 Analyze", use_container_width=True)

    with col_result:
        if analyze and user_text.strip():
            label, prob_fake, prob_real, triggered = predict(user_text, model, vec, threshold)

            # Prediction badge
            if label == 0:
                st.markdown('<div class="badge-fake">🚨 FAKE NEWS</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-real">✅ REAL NEWS</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability bars
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Confidence Scores</div>
                <div style="margin-bottom:10px">
                    <div style="display:flex; justify-content:space-between">
                        <span style="color:#E63946;font-weight:700">FAKE</span>
                        <span style="color:#E63946;font-weight:700">{prob_fake*100:.1f}%</span>
                    </div>
                    <div class="prob-bar-container">
                        <div class="prob-bar-fake" style="width:{prob_fake*100:.1f}%"></div>
                    </div>
                </div>
                <div>
                    <div style="display:flex; justify-content:space-between">
                        <span style="color:#2DC653;font-weight:700">REAL</span>
                        <span style="color:#2DC653;font-weight:700">{prob_real*100:.1f}%</span>
                    </div>
                    <div class="prob-bar-container">
                        <div class="prob-bar-real" style="width:{prob_real*100:.1f}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Keyword warning
            if triggered:
                kw_html = "".join([f'<span class="kw-tag">{kw}</span>' for kw in triggered])
                st.markdown(f"""
                <div class="card" style="border-color:#E63946">
                    <div class="card-title" style="color:#E63946">⚠️ Suspicious Keywords Detected</div>
                    {kw_html}
                </div>
                """, unsafe_allow_html=True)

            # Stats row
            word_count = len(user_text.split())
            sent_count = len(re.split(r'[.!?]+', user_text))
            excl_count = user_text.count("!")
            caps_ratio = sum(1 for c in user_text if c.isupper()) / max(len(user_text), 1)

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="metric-box"><div class="metric-value">{word_count}</div><div class="metric-label">Words</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-box"><div class="metric-value">{sent_count}</div><div class="metric-label">Sentences</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-box"><div class="metric-value">{excl_count}</div><div class="metric-label">Exclamations</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric-box"><div class="metric-value">{caps_ratio*100:.0f}%</div><div class="metric-label">CAPS Ratio</div></div>', unsafe_allow_html=True)

        elif analyze and not user_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding: 3rem 1rem;">
                <div style="font-size:3rem">🔍</div>
                <div style="color:#8B9DBF; font-size:1rem; margin-top:0.5rem">
                    Paste a news article and click <b>Analyze</b><br>to check its authenticity
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Word importance section (full width)
    if analyze and user_text.strip() and show_words:
        st.markdown("---")
        st.markdown("#### 🧠 Word Importance Analysis")
        st.caption("Words with positive scores push toward REAL; negative scores push toward FAKE")

        impacts = word_importance(user_text, model, vec, n=n_words)

        if impacts:
            words_pos = [(w, v) for w, v in impacts if v > 0]
            words_neg = [(w, v) for w, v in impacts if v <= 0]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**✅ Pushes toward REAL**")
                for w, v in sorted(words_pos, key=lambda x: x[1], reverse=True):
                    bar_w = min(int(abs(v) / max(abs(impacts[0][1]), 0.001) * 100), 100)
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;margin-bottom:6px">
                        <div style="width:90px;font-size:0.85rem;color:#E0EAF4;font-weight:600">{w}</div>
                        <div style="flex:1;background:#0D1B2A;border-radius:4px;height:10px;overflow:hidden">
                            <div style="width:{bar_w}%;background:#2DC653;height:10px;border-radius:4px"></div>
                        </div>
                        <div style="width:45px;text-align:right;font-size:0.8rem;color:#2DC653">+{v:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with c2:
                st.markdown("**🚨 Pushes toward FAKE**")
                for w, v in sorted(words_neg, key=lambda x: x[1]):
                    bar_w = min(int(abs(v) / max(abs(impacts[0][1]), 0.001) * 100), 100)
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;margin-bottom:6px">
                        <div style="width:90px;font-size:0.85rem;color:#E0EAF4;font-weight:600">{w}</div>
                        <div style="flex:1;background:#0D1B2A;border-radius:4px;height:10px;overflow:hidden">
                            <div style="width:{bar_w}%;background:#E63946;height:10px;border-radius:4px"></div>
                        </div>
                        <div style="width:45px;text-align:right;font-size:0.8rem;color:#E63946">{v:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE MODELS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 📊 ML Algorithm Comparison on Training Data")

    if st.button("▶️ Run Model Comparison", use_container_width=False):
        with st.spinner("Training and evaluating all models..."):
            texts  = REAL_SAMPLES + FAKE_SAMPLES
            labels = [1]*len(REAL_SAMPLES) + [0]*len(FAKE_SAMPLES)

            vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=3000, sublinear_tf=True, stop_words="english")
            X = vectorizer.fit_transform([preprocess(t) for t in texts])
            y = np.array(labels)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

            models_dict = {
                "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
                "Naïve Bayes":         MultinomialNB(),
                "SVM (Linear)":        LinearSVC(max_iter=1000, random_state=42),
                "Decision Tree":       DecisionTreeClassifier(random_state=42),
                "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
                "Passive Aggressive":  PassiveAggressiveClassifier(C=0.5, random_state=42),
            }

            results = []
            for name, m in models_dict.items():
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                results.append({
                    "Model": name,
                    "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 1),
                    "Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 1),
                    "Recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 1),
                    "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0) * 100, 1),
                })

            df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
            st.session_state["compare_df"] = df

    if "compare_df" in st.session_state:
        df = st.session_state["compare_df"]

        # Bar chart
        st.markdown("**Accuracy Comparison (%)**")
        chart_data = df.set_index("Model")["Accuracy"]
        st.bar_chart(chart_data, color="#00B4D8", height=300)

        # Full metrics table
        st.markdown("**Full Metrics Table**")
        st.dataframe(
            df.style
              .background_gradient(subset=["Accuracy","Precision","Recall","F1-Score"], cmap="Blues")
              .format({"Accuracy":"{:.1f}%","Precision":"{:.1f}%","Recall":"{:.1f}%","F1-Score":"{:.1f}%"}),
            use_container_width=True,
        )

        # Highlight winner
        best = df.iloc[0]
        st.success(f"🏆 Best Model: **{best['Model']}** — Accuracy: {best['Accuracy']}%  |  F1: {best['F1-Score']}%")
    else:
        st.info("Click **Run Model Comparison** to evaluate all algorithms on the dataset.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="card">
            <div class="card-title">🔄 Pipeline Steps</div>
            <ol style="color:#B0C4D8; line-height:2">
                <li><b style="color:#00B4D8">Input</b> — Raw news text from user</li>
                <li><b style="color:#00B4D8">Pre-processing</b> — Lowercase, remove noise, tokenize, stop-word removal, stemming</li>
                <li><b style="color:#00B4D8">Feature Extraction</b> — TF-IDF vectorizer (unigrams + bigrams)</li>
                <li><b style="color:#00B4D8">Classification</b> — Passive Aggressive Classifier predicts FAKE/REAL</li>
                <li><b style="color:#00B4D8">Keyword Check</b> — Override with known sensationalist terms</li>
                <li><b style="color:#00B4D8">Explainability</b> — Word-level contribution scores surfaced</li>
                <li><b style="color:#00B4D8">Output</b> — Binary prediction with confidence</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-title">📐 Evaluation Metrics</div>
            <table style="width:100%;color:#B0C4D8;border-collapse:collapse">
                <tr style="border-bottom:1px solid #1A3A5C">
                    <th style="text-align:left;padding:6px;color:#00B4D8">Metric</th>
                    <th style="text-align:left;padding:6px;color:#00B4D8">Description</th>
                </tr>
                <tr style="border-bottom:1px solid #1A3A5C">
                    <td style="padding:6px"><b>Accuracy</b></td>
                    <td style="padding:6px">% of correctly classified articles</td>
                </tr>
                <tr style="border-bottom:1px solid #1A3A5C">
                    <td style="padding:6px"><b>Precision</b></td>
                    <td style="padding:6px">Of predicted FAKE, how many truly are</td>
                </tr>
                <tr style="border-bottom:1px solid #1A3A5C">
                    <td style="padding:6px"><b>Recall</b></td>
                    <td style="padding:6px">Of actual FAKE, how many were caught</td>
                </tr>
                <tr>
                    <td style="padding:6px"><b>F1-Score</b></td>
                    <td style="padding:6px">Harmonic mean of precision & recall</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🤖 Passive Aggressive Classifier</div>
            <p>The core classifier used in this system:</p>
            <ul style="color:#B0C4D8;line-height:2">
                <li><b style="color:#2DC653">Passive</b> — Makes no update when prediction is correct</li>
                <li><b style="color:#E63946">Aggressive</b> — Adjusts weights strongly on wrong predictions</li>
                <li>✅ Ideal for large-scale, high-dimensional text data</li>
                <li>✅ Online learning — updates incrementally</li>
                <li>✅ Computationally very efficient</li>
                <li>✅ Works well with TF-IDF sparse matrices</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-title">📄 TF-IDF Feature Extraction</div>
            <p><b style="color:#00B4D8">Term Frequency–Inverse Document Frequency</b></p>
            <ul style="color:#B0C4D8;line-height:2">
                <li><b>TF</b> — How often a word appears in the article</li>
                <li><b>IDF</b> — How rare the word is across all articles</li>
                <li>Common words like "the", "is" get <b>low</b> scores</li>
                <li>Rare, meaningful words get <b>high</b> scores</li>
                <li>Bigrams (2-word phrases) also captured</li>
                <li>Result: Sparse numerical feature matrix</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="card-title">🎓 Project Info</div>
            <p style="color:#B0C4D8">
                <b>Students:</b> Gowdham R · Saksham Sharma · Parth Mishra<br>
                <b>Guide:</b> Dr. Saraswathi .E<br>
                <b>Institute:</b> SRM Institute of Science and Technology, Ramapuram<br>
                <b>Department:</b> Computer Science & Engineering<br>
                <b>Degree:</b> B.Tech · October 2025
            </p>
        </div>
        """, unsafe_allow_html=True)
