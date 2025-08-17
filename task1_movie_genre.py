# Movie Genre Classification - CODSOFT Internship
import warnings
warnings.filterwarnings("ignore")   # 🚀 This hides the UndefinedMetricWarning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (download from Kaggle: hijest/genre-classification-dataset-imdb)
df = pd.read_csv("train_data.txt", sep=":::", engine="python", header=None)
df.columns = ["id", "title", "genre", "description"]

# Features & labels
X = df["description"]
y = df["genre"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
print("📌 Movie Genre Classification Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
