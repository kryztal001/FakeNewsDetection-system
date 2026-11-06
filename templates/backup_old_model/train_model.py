
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. LOAD DATASETS
# ==========================================

print("Loading datasets...")

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

print(f"Fake news records: {len(fake)}")
print(f"Real news records: {len(true)}")


# ==========================================
# 2. ADD LABELS
# ==========================================

# 0 = Fake News
# 1 = Real News

fake["label"] = 0
true["label"] = 1


# ==========================================
# 3. COMBINE DATASETS
# ==========================================

data = pd.concat([fake, true], ignore_index=True)


# ==========================================
# 4. KEEP ONLY REQUIRED COLUMNS
# ==========================================

data = data[["text", "label"]]


# ==========================================
# 5. REMOVE MISSING VALUES
# ==========================================

data.dropna(inplace=True)


# ==========================================
# 6. REMOVE DUPLICATE ARTICLES
# ==========================================

before = len(data)

data.drop_duplicates(subset=["text"], inplace=True)

after = len(data)

print(f"Removed duplicates: {before - after}")
print(f"Total records after cleaning: {after}")


# ==========================================
# 7. SHUFFLE DATA
# ==========================================

data = data.sample(frac=1, random_state=42).reset_index(drop=True)


# ==========================================
# 8. SEPARATE TEXT AND LABEL
# ==========================================

X = data["text"]
y = data["label"]


# ==========================================
# 9. SPLIT DATA BEFORE TF-IDF
# ==========================================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 10. CREATE TF-IDF VECTORIZER
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)


# ==========================================
# 11. FIT TF-IDF ONLY ON TRAINING DATA
# ==========================================

print("Converting text to TF-IDF features...")

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# ==========================================
# 12. TRAIN MODEL
# ==========================================

print("Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# 13. TEST MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print()
print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Accuracy: {accuracy * 100:.2f}%")

print()
print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Fake News", "Real News"]
))

print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 14. SAVE MODEL
# ==========================================

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print()
print("=" * 50)
print("Training completed successfully!")
print("Model saved as: model.pkl")
print("Vectorizer saved as: vectorizer.pkl")
print("=" * 50)

