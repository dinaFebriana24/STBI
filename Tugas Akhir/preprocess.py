import os
import pandas as pd
import torch
import clip
import faiss
import pickle
from PIL import Image
import kagglehub
import numpy as np
import gc


# Download dataset
data_path = kagglehub.dataset_download('paramaggarwal/fashion-product-images-dataset')
styles_path = os.path.join(data_path, 'fashion-dataset', 'styles.csv')
images_folder = os.path.join(data_path, 'fashion-dataset', 'images')

# Load data
styles_df = pd.read_csv(styles_path, on_bad_lines='skip')
styles_df['id'] = styles_df['id'].astype(str).str.strip().str.lower()
styles_df = styles_df.dropna(subset=['productDisplayName'])

# Limit the dataset to a smaller subset for testing
styles_df = styles_df.head(400)

# Filter valid images
styles_df['image_path'] = styles_df['id'].apply(lambda x: os.path.join(images_folder, x + '.jpg'))
styles_df = styles_df[styles_df['image_path'].apply(os.path.isfile)]

# Setup CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Helper functions
def get_clip_text_embeddings(texts):
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
    return text_features.cpu().numpy()

def get_clip_image_embeddings(image_path):
    image = Image.open(image_path).convert("RGB")
    image = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    return image_features.cpu().numpy()

# Compute embeddings
def safe_get_text_embeddings(row):
    try:
        return get_clip_text_embeddings([row])
    except Exception as e:
        print(f"Error generating text embedding for: {row}, Error: {e}")
        return np.array([])

def safe_get_image_embeddings(image_path):
    try:
        return get_clip_image_embeddings(image_path)
    except Exception as e:
        print(f"Error generating image embedding for: {image_path}, Error: {e}")
        return np.array([])

# Process embeddings in batches
batch_size = 5  # Adjust as needed

def compute_embeddings_in_batches(df, batch_size, embedding_function):
    embeddings = []
    for start_idx in range(0, len(df), batch_size):
        batch = df.iloc[start_idx:start_idx+batch_size]
        embeddings_batch = batch['productDisplayName'].apply(safe_get_text_embeddings)
        embeddings.extend(embeddings_batch)
        del batch, embeddings_batch
        gc.collect()
    return embeddings

styles_df['text_embeddings'] = compute_embeddings_in_batches(styles_df, batch_size, safe_get_text_embeddings)
styles_df['image_embeddings'] = compute_embeddings_in_batches(styles_df, batch_size, safe_get_image_embeddings)

# Filter invalid embeddings
styles_df = styles_df[
    styles_df['text_embeddings'].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0) &
    styles_df['image_embeddings'].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)
]

# Create FAISS index
text_embeddings_matrix = np.vstack(styles_df['text_embeddings'].values)
image_embeddings_matrix = np.vstack(styles_df['image_embeddings'].values)

# Use a smaller FAISS index (IVF)
nlist = 5
quantizer = faiss.IndexFlatL2(text_embeddings_matrix.shape[1])
text_index = faiss.IndexIVFFlat(quantizer, text_embeddings_matrix.shape[1], nlist, faiss.METRIC_L2)
image_index = faiss.IndexIVFFlat(quantizer, image_embeddings_matrix.shape[1], nlist, faiss.METRIC_L2)

# Train FAISS index
text_index.train(text_embeddings_matrix)
image_index.train(image_embeddings_matrix)

# Add embeddings to the FAISS index
text_index.add(text_embeddings_matrix)
image_index.add(image_embeddings_matrix)

# Save FAISS index
faiss.write_index(text_index, "preprocessed/text_index.faiss")
faiss.write_index(image_index, "preprocessed/image_index.faiss")

print("Preprocessing complete. Files saved to 'preprocessed/'")


