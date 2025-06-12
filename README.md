
## 📰 Fake News Detector using NLP and Machine Learning

### 📌 Overview

This project is a simple and effective **Fake News Detection system** built using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. It classifies short news texts as **real** or **fake** based on how the language is used.

---

### 🎯 Objective

To create a Python-based application that:

* Preprocesses news headlines/articles
* Extracts features using TF-IDF
* Trains a machine learning model (Naive Bayes)
* Predicts whether new input news is real or fake

---

### 🛠 Technologies Used

| Area          | Tools                   |
| ------------- | ----------------------- |
| Language      | Python                  |
| NLP           | NLTK                    |
| ML            | scikit-learn            |
| Data Handling | pandas, numpy           |
| Model         | Multinomial Naive Bayes |
| Vectorization | TF-IDF                  |

---

### 🧠 Workflow

1. **Dataset Creation**
   A small sample dataset is used (can be expanded to real datasets like those from Kaggle).

2. **Text Preprocessing**

   * Removal of special characters
   * Conversion to lowercase
   * Stopword removal
   * Stemming using PorterStemmer

3. **Feature Extraction**
   Text is converted to numerical features using **TF-IDF Vectorizer**.

4. **Model Training**
   Trained using the **Multinomial Naive Bayes** classifier.

5. **Model Evaluation**
   Accuracy score and classification report show how well the model performs.

6. **Prediction Loop**
   A command-line loop allows users to input any news headline and get an instant prediction.

---

### 📦 Folder Structure

```
FakeNewsDetector/
├── fakenews.py             # Main code file
├── README.md               # Project description
├── requirements.txt        # List of dependencies
└── .gitignore              # Optional: Python cache exclusion
```

---

### 📋 Sample Output

```
✅ Accuracy: 100.0%
✅ Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         1

Enter a news text (or type 'exit' to quit): NASA discovers alien base on Mars!
📰 Prediction: Fake News
```

---

### 📈 Future Enhancements

* Use real-world datasets (e.g. from Kaggle)
* Build a GUI using Tkinter
* Deploy as a web app using Flask or Streamlit
* Add deep learning models (e.g., LSTM)
* Use word embeddings (e.g., Word2Vec or BERT)

---

### 🧾 Requirements

Install the required libraries using:

```bash
pip install -r requirements.txt
```

**requirements.txt:**

```
pandas
numpy
nltk
scikit-learn
```

---

### 🙌 Credits

Created by Cheran G as part of a machine learning mini project to explore how natural language processing can help fight misinformation.
