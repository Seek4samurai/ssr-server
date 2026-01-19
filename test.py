# Testing file
# Nothing necessary

from openai import OpenAI
import pandas as pd
import numpy as np
import pickle
import torch

from sentence_transformers import SentenceTransformer

print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))


model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv("./merged_song_dataset.csv")


# def embed_text(text):
#     resp = client.embeddings.create(model="text-embedding-3-small", input=text)
#     return resp.data[0].embedding


df["semantic_text"] = (
    df["track_name_x"].astype(str) + " " + df["album_name_x"].astype(str)
)

# Generate embeddings
# df["embedding"] = df["semantic_text"].apply(lambda x: embed_text(x))

df["embedding"] = df["semantic_text"].apply(lambda x: model.encode(x).tolist())

with open("./spotify_embeddings.pkl", "wb") as f:
    pickle.dump(df, f)
