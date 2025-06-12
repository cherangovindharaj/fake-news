# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK stopwords (only the first time you run)
nltk.download('stopwords')

# Step 1: Sample Dataset
data = {
    'text': [
        "The Prime Minister has announced new economic reforms today.",
        "Aliens have landed on Earth, confirms secret NASA files!",
        "Scientists found a cure for cancer in mushrooms.",
        "Click here to win a free iPhone. Limited offer!",
        "The government passed the education bill in parliament.",
        "Celebrity cloned! Shocking truth about famous actor revealed!",
        "New policy introduced to reduce pollution in major cities.",
        "You won't believe what this dog did to become a millionaire!"
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Real, 0 = Fake
}

# Step 2: Create a DataFrame
df = pd.DataFrame(data)

# Step 3: Clean the Text
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove symbols
    text = text.lower().split()            # Convert to lowercase and split
    words = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# Step 4: Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['label']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 6: Train the Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Predict News from User Input
def predict_news(news):
    cleaned = clean_text(news)
    vector = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]
    return "Real News" if prediction == 1 else "Fake News"

# Step 9: Loop for User Input
while True:
    user_input = input("\nEnter a news text (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print("📰 Prediction:", predict_news(user_input))
