# 🔍 Fake News Detection System
**SRM Institute of Science and Technology — B.Tech CSE Project**

**Team:** Gowdham R · Saksham Sharma · Parth Mishra  
**Guide:** Dr. Saraswathi .E  

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📁 Project Structure
```
fake_news_project/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Detect Tab** | Paste any news article to classify as FAKE or REAL |
| 📊 **Compare Models Tab** | Run & compare 7 ML algorithms live |
| 📖 **How It Works Tab** | Pipeline explanation & metric definitions |
| 🧠 **Word Importance** | See which words push toward FAKE vs REAL |
| ⚠️ **Keyword Detection** | Automatic flagging of known fake-news terms |
| 🎛️ **Sensitivity Slider** | Adjust fake detection threshold |

---

## 🤖 Tech Stack
- **Framework:** Streamlit
- **ML:** scikit-learn (Passive Aggressive Classifier)
- **Features:** TF-IDF Vectorizer (unigrams + bigrams)
- **Explainability:** Word-level contribution scores

---

## 📊 Model Performance
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| FAKE | 0.91 | 0.86 | 0.88 |
| REAL | 0.87 | 0.92 | 0.90 |
| **Overall Accuracy** | | | **~89%** |
