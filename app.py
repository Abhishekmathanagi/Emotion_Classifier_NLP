
import pickle

import streamlit as st

from emotion_preprocessing import clean_text, ensure_nltk_data

st.set_page_config(page_title="Emotion Classifier", page_icon="🎭", layout="centered")

st.markdown(
    """
    <style>
    .main > div { padding-top: 2rem; }
    .emotion-result {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .emotion-emoji { font-size: 3rem; }
    .emotion-label { font-size: 1.6rem; font-weight: 700; margin-top: .25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    ensure_nltk_data()
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("emotion_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    return tfidf, model, label_map


EMOTION_STYLE = {
    "joy":      {"emoji": "😄", "color": "#FFF3CD"},
    "happy":    {"emoji": "😄", "color": "#FFF3CD"},
    "sadness":  {"emoji": "😢", "color": "#D6E4F0"},
    "anger":    {"emoji": "😠", "color": "#F8D7DA"},
    "fear":     {"emoji": "😨", "color": "#E2D9F3"},
    "love":     {"emoji": "❤️", "color": "#FADCE4"},
    "surprise": {"emoji": "😲", "color": "#D1ECF1"},
}
DEFAULT_STYLE = {"emoji": "🔎", "color": "#E9ECEF"}

# ------------------------------------------------------------------- header
st.title("🎭 Emotion Classifier")
st.caption("TF-IDF + Logistic Regression · trained on train.txt")
st.write("Type a sentence and the model will predict the emotion behind it.")

try:
    tfidf, model, label_map = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Couldn't find a required file ({e.filename}). Make sure "
        "`tfidf_vectorizer.pkl`, `emotion_model.pkl`, `label_map.pkl` and "
        "`preprocessing.py` are all in the same folder as `app.py`."
    )
    st.stop()


with st.sidebar:
    st.header("About")
    st.write(
        "This app cleans your text (lowercasing, removing punctuation, "
        "digits, URLs, HTML tags, non-ASCII characters and stopwords), "
        "then vectorizes it with TF-IDF and classifies the emotion with "
        "a Logistic Regression model."
    )
    st.divider()
    st.subheader("Known emotions")
    for emo in label_map.values():
        style = EMOTION_STYLE.get(emo.lower(), DEFAULT_STYLE)
        st.write(f"{style['emoji']} {emo}")

text_input = st.text_area(
    "Your text",
    placeholder="e.g. I can't believe I got the job, I'm so excited!",
    height=120,
)

col1, col2 = st.columns([1, 1])
with col1:
    predict_clicked = st.button("Predict emotion", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if predict_clicked:
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(text_input)

        if not cleaned.strip():
            st.warning("Nothing was left to analyze after cleaning — try a different sentence.")
        else:
            vec = tfidf.transform([cleaned])

            pred_num = model.predict(vec)[0]
            pred_label = label_map[pred_num]
            style = EMOTION_STYLE.get(pred_label.lower(), DEFAULT_STYLE)

            st.markdown(
                f"""
                <div class="emotion-result" style="background-color:{style['color']};">
                    <div class="emotion-emoji">{style['emoji']}</div>
                    <div class="emotion-label">{pred_label.capitalize()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vec)[0]
                probs = {label_map[i]: float(p) for i, p in enumerate(proba)}
                probs = dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))

                st.subheader("Confidence by emotion")
                for emo, p in probs.items():
                    st.write(f"{EMOTION_STYLE.get(emo.lower(), DEFAULT_STYLE)['emoji']} {emo.capitalize()}")
                    st.progress(p, text=f"{p:.1%}")

            with st.expander("Show cleaned text used for prediction"):
                st.code(cleaned)

st.divider()
st.caption("Built with Streamlit · scikit-learn TF-IDF + Logistic Regression")