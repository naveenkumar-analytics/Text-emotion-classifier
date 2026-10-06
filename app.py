import os
import pickle
 
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
# ------------------ CONFIG (must match training) ------------------
MAX_LEN = 50
MODEL_PATH = os.path.join("Artifacts", "BiGRU_Model.keras")
TOKENIZER_PATH = os.path.join("Artifacts", "tokenizer.pkl")
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOJI = {"sadness": "😢", "joy": "😄", "love": "❤️", "anger": "😠", "fear": "😨", "surprise": "😲"}
 
# EDIT: paste your own test results here after training (use None if unknown)
RESULTS = {
    "Model": ["Simple RNN", "LSTM", "GRU", "Bidirectional GRU"],
    "Test Accuracy (%)": [None, None, None, None],
}
 
EXAMPLES = {
    "Happy": "I can't believe how happy I am right now, this is amazing!",
    "Sad": "I feel so alone and hopeless today.",
    "Angry": "I am furious that they cancelled the trip at the last minute.",
    "Scared": "I feel terrified when walking down dark alleyways alone.",
}
 
st.set_page_config(page_title="Emotion Classifier | NLP Project", page_icon="🧠", layout="wide")
 
# ------------------ STYLE ------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 14px;
           background: linear-gradient(135deg, #4f46e5, #06b6d4); color: white; margin-bottom: 1.2rem;}
    .hero h1 {margin: 0; font-size: 2rem;}
    .hero p {margin: .3rem 0 0 0; opacity: .95;}
    .result {padding: 1.2rem; border-radius: 12px; text-align: center;
             border: 1px solid rgba(128,128,128,.3);}
    .result .big {font-size: 3rem; margin: 0;}
    .result .name {font-size: 1.6rem; font-weight: 700; text-transform: capitalize;}
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
# ------------------ LOAD MODEL (cached) ------------------
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer
 
 
def predict(text, model, tokenizer):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="pre", truncating="post")  # same as training
    return model.predict(padded, verbose=0)[0]
 
 
# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("👤 About the Developer")
    st.markdown(
        "**Naveen Kumar**  \nAspiring Data Scientist / ML Engineer  \n"
        "[GitHub](https://github.com/naveenkumar-analytics) · "
        "[LinkedIn](https://www.linkedin.com/in/naveen-kumarofficial/)"
    )
    st.divider()
    st.header("🛠 Tech Stack")
    st.markdown("Python · TensorFlow/Keras · NLP · Pandas · Scikit-learn · Streamlit")
    st.divider()
    st.caption("Dataset: dair-ai/emotion (16K train / 2K val / 2K test)")
 
# ------------------ HERO ------------------
st.markdown(
    """
    <div class="hero">
      <h1>🧠 Emotion Classification from Text</h1>
      <p>Deep Learning NLP project using a Bidirectional GRU to detect 6 emotions:
      sadness, joy, love, anger, fear, surprise.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
try:
    model, tokenizer = load_artifacts()
except Exception as e:
    st.error(
        "Model files not found. Run `emotion_classification.py` first so that the "
        "`Artifacts/` folder is created, then restart this app."
    )
    st.exception(e)
    st.stop()
 
tab1, tab2, tab3 = st.tabs(["🔮 Try the Model", "📊 Model Performance", "📘 Project Details"])
 
# ------------------ TAB 1: PREDICT ------------------
with tab1:
    if "text" not in st.session_state:
        st.session_state.text = ""
 
    st.markdown("**Quick examples:**")
    cols = st.columns(len(EXAMPLES))
    for col, (name, sentence) in zip(cols, EXAMPLES.items()):
        if col.button(name, use_container_width=True):
            st.session_state.text = sentence
 
    user_text = st.text_area("Enter a sentence:", key="text", height=110,
                             placeholder="e.g. I feel so excited about my new job!")
 
    if st.button("Analyze Emotion", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            probs = predict(user_text, model, tokenizer)
            top = int(np.argmax(probs))
            label = LABELS[top]
 
            left, right = st.columns([1, 2])
            with left:
                st.markdown(
                    f'<div class="result"><p class="big">{EMOJI[label]}</p>'
                    f'<p class="name">{label}</p>'
                    f'<p>Confidence: <b>{probs[top]*100:.1f}%</b></p></div>',
                    unsafe_allow_html=True,
                )
            with right:
                chart_df = pd.DataFrame({"Emotion": LABELS, "Probability": probs}).set_index("Emotion")
                st.bar_chart(chart_df)
 
# ------------------ TAB 2: PERFORMANCE ------------------
with tab2:
    st.subheader("Model Comparison")
    df = pd.DataFrame(RESULTS)
    if df["Test Accuracy (%)"].notna().any():
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Add your test accuracies in the `RESULTS` dictionary at the top of `app.py`.")
    st.markdown(
        "- **Loss:** sparse categorical cross-entropy (classification, not MSE)\n"
        "- **Optimizer:** Adam\n"
        "- **Imbalance handling:** class weights\n"
        "- **Regularization:** Dropout (0.5) + EarlyStopping on validation loss"
    )
 
# ------------------ TAB 3: DETAILS ------------------
with tab3:
    st.subheader("Pipeline")
    st.markdown(
        """
        1. **Data:** dair-ai/emotion from Hugging Face (train / validation / test splits)
        2. **Preprocessing:** Keras Tokenizer (10,000 words, OOV token), pre-padding to 50 tokens
        3. **Models compared:** Simple RNN, LSTM, GRU, Bidirectional GRU
        4. **Training:** class-weighted loss, validation-based early stopping (no test leakage)
        5. **Evaluation:** accuracy, confusion matrix, per-class precision/recall/F1
        6. **Deployment:** Streamlit web app with cached model loading
        """
    )
    st.subheader("Key Learnings")
    st.markdown(
        "- Padding side matters for unidirectional RNNs (pre-padding fixed poor accuracy)\n"
        "- Bidirectional layers capture context from both directions\n"
        "- Keep the test set untouched until final evaluation"
    )
