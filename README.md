# 🧠 Text Emotion Classifier (NLP + Deep Learning)

A deep learning project that detects **6 emotions** (sadness, joy, love, anger, fear, surprise) from English text.
Four recurrent architectures were trained and compared, and the best model (Bidirectional GRU) is deployed in an interactive **Streamlit** web app.

🔗 **Live Demo:**https://text-emotion-classifier-3k9qvmoqupsl9ypwjdhyym.streamlit.app/



## Features
- Real-time emotion prediction with confidence score
- Probability chart for all 6 emotions
- One-click example sentences
- Model comparison and project details tabs

## Tech Stack
Python · TensorFlow · Keras · NLP · Pandas · NumPy · Scikit-learn · Seaborn · Matplotlib · Streamlit · Hugging Face Datasets

## Dataset
[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion): 16,000 train / 2,000 validation / 2,000 test samples, 6 emotion classes.

## Methodology
1. **EDA:** class distribution, missing value check
2. **Preprocessing:** Keras Tokenizer (10,000 words, OOV token), pre-padding to length 50
3. **Models:** Simple RNN, LSTM, GRU, Bidirectional GRU
4. **Training:** Adam optimizer, sparse categorical cross-entropy, class weights for imbalance, Dropout, EarlyStopping on a separate validation set (no test leakage)
5. **Evaluation:** accuracy, confusion matrix (raw + normalized), precision/recall/F1
6. **Deployment:** Streamlit app with cached model loading

## Results
| Model | Test Accuracy |
|---|---|
| Simple RNN | 77.85% |
| LSTM | 91.70% |
| GRU | 92.45% |
| **Bidirectional GRU** | **92.45%** |

## Key Learnings
- Padding side matters for unidirectional RNNs: pre-padding fixed very low accuracy
- Bidirectional layers capture context from both directions
- Keep the test set untouched until final evaluation

## Project Structure
```
Text-emotion-classifier/
├── Artifacts/
│   ├── BiGRU_Model.keras         # trained model
│   └── tokenizer.pkl             # fitted tokenizer
├── .gitignore
├── README.md
├── app.py                        # Streamlit UI
├── emotion_classification.ipynb  # training and evaluation notebook
└── requirements.txt
```

## Run Locally
```bash
git clone https://github.com/naveenkumar-analytics/Text-emotion-classifier.git
cd Text-emotion-classifier
pip install -r requirements.txt
streamlit run app.py
```
To retrain, run all cells in `emotion_classification.ipynb`; it recreates the `Artifacts/` folder.

## Author
**Naveen Kumar** · [LinkedIn](https://www.linkedin.com/in/naveen-kumarofficial/) · [GitHub](https://github.com/naveenkumar-analytics)
