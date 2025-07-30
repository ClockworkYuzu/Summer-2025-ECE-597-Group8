import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import nltk

nltk.download('punkt')

file_name = 'spam_ham_dataset_cleaned'
# 1. Load cleaned data
df = pd.read_csv(f"../data/{file_name}.csv")
df["clean_text"] = df["clean_text"].fillna("")

# 2. Tokenize by splitting clean_text
df["tokens"] = df["clean_text"].str.split()


# ### Word Embedding - Word2Vec

# 3. Train Word2Vec model
w2v_model = Word2Vec(
    sentences=df["tokens"].tolist(),
    vector_size=300,
    window=5,
    min_count=2,
    workers=4,
    epochs=10
)

# 4. Create document embedding by averaging word vectors
def avg_embedding(tokens):
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)
df["w2v_embedding"] = df["tokens"].apply(avg_embedding)

# 5. Expand embeddings into separate columns
w2v_cols = [f"w2v_{i}" for i in range(w2v_model.vector_size)]
w2v_df = pd.DataFrame(df["w2v_embedding"].tolist(), columns=w2v_cols)
result = pd.concat([df.drop(columns=["w2v_embedding"]), w2v_df], axis=1)

# 6. Save to new CSV - check and it's fine 
w2v_df.to_csv("../features/w2v_embeddings.csv", index=False)


# ### Word Embedding - BERT

import pandas as pd
from sentence_transformers import SentenceTransformer

# 1. Load cleaned data
texts = df["clean_text"].tolist()
#texts = df["clean_text"].tolist()
# 2. Load pre-trained BERT model (SentenceTransformer)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Compute embeddings for each email
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

# 4. Expand embeddings into DataFrame
bert_cols = [f"bert_{i}" for i in range(embeddings.shape[1])]
bert_df = pd.DataFrame(embeddings, columns=bert_cols)
result = pd.concat([df.reset_index(drop=True), bert_df], axis=1)

# 5. Save to new CSV
bert_df.to_csv("../features/bert_embeddings.csv", index=False)


