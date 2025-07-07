import pandas as pd
import numpy as np
import re
import string
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Preprocess the text
def clean_text(raw_text):
    # Remove HTML
    text = BeautifulSoup(raw_text, "html.parser").get_text()
    # Remove URLs
    text = re.sub(r"http\\S+|www\\.\\S+", "", text)
    # Lowercase
    text = text.lower()
    # Remove digits and punctuation
    text = re.sub(r"[\\d]+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Simple split as tokenization (no NLTK)
    tokens = text.split()
    # Remove stopwords manually
    stop_words = set([
        'the', 'and', 'is', 'in', 'to', 'of', 'that', 'this', 'on', 'for', 'it', 'with',
        'as', 'was', 'but', 'are', 'at', 'by', 'a', 'an', 'be', 'or', 'from', 'has', 'have', 'you'
    ])
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# Construct features manually
def extract_features(row):
    raw = row["text"]
    return pd.Series({
        "char_count": len(raw),
        "word_count": len(raw.split()),
        "uppercase_ratio": sum(1 for c in raw if c.isupper()) / max(len(raw), 1),
        "num_urls": len(re.findall(r"http\S+|www.\S+", raw)),
        "num_exclamations": raw.count("!"),
    })

file_name = 'spam_ham_dataset'
# 1. Read data
df = pd.read_csv(f"../data/{file_name}.csv")
# df = df.drop(columns=["Unnamed: 2", "Unnamed: 3"])
df["Subject"] = df["Subject"].fillna("")
df["text"] = df["Subject"] + " " + df["Body"]
df["text"] = df["text"].fillna("")

# 2. Clean text
df["clean_text"] = df["text"].apply(clean_text)
df.to_csv(f"../data/{file_name}_cleaned.csv", index=False)

# 3. Construct features manually
features = df.apply(extract_features, axis=1)

# 4. TF-IDF features
tfidf = TfidfVectorizer(max_features=300)
tfidf_matrix = tfidf.fit_transform(df["clean_text"])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# 5. Combine features
X = pd.concat([features.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

# 6. Extract Labels
y = df['Label']

# 7. Save the data
X.to_csv("../features/features.csv", index=False)
pd.DataFrame({"label": y}).to_csv("../features/labels.csv", index=False)

print("✅ Feature extration done，save to features.csv")