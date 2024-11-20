from flask import Flask, render_template, request
import os
import pandas as pd
from PIL import Image
import torch
import kagglehub
import clip
import faiss
import matplotlib.pyplot as plt

# Setup Flask
app = Flask(__name__)

# Load data
data_path = kagglehub.dataset_download('paramaggarwal/fashion-product-images-dataset')
styles_path = os.path.join(data_path, 'fashion-dataset', 'styles.csv')
images_folder = os.path.join(data_path, 'fashion-dataset', 'images')

styles_df = pd.read_csv(styles_path, on_bad_lines='skip')
styles_df['id'] = styles_df['id'].astype(str).str.strip().str.lower()
styles_df = styles_df.dropna(subset=['productDisplayName'])

styles_df['image_path'] = styles_df['id'].apply(lambda x: os.path.join(images_folder, x + '.jpg'))
styles_df = styles_df[styles_df['image_path'].apply(os.path.isfile)]

# Prepare CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

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

# Generate embeddings for the dataset
styles_df['text_embeddings'] = styles_df['productDisplayName'].apply(lambda x: get_clip_text_embeddings([x]))
styles_df['image_embeddings'] = styles_df['image_path'].apply(get_clip_image_embeddings)

# Build FAISS index
text_embeddings_matrix = np.vstack(styles_df['text_embeddings'].values)
image_embeddings_matrix = np.vstack(styles_df['image_embeddings'].values)

text_index = faiss.IndexFlatL2(text_embeddings_matrix.shape[1])
text_index.add(text_embeddings_matrix)

image_index = faiss.IndexFlatL2(image_embeddings_matrix.shape[1])
image_index.add(image_embeddings_matrix)

# Search functions
def search_similar_text(query, k=5):
    query_embedding = get_clip_text_embeddings([query])
    distances, indices = text_index.search(query_embedding, k)
    return styles_df.iloc[indices[0]]

def search_similar_image(image_path, k=5):
    query_embedding = get_clip_image_embeddings(image_path)
    distances, indices = image_index.search(query_embedding, k)
    return styles_df.iloc[indices[0]]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query_type = request.form["query_type"]
    results = None

    if query_type == "text":
        query_text = request.form["query_text"]
        results = search_similar_text(query_text)
    elif query_type == "image":
        uploaded_file = request.files["query_image"]
        if uploaded_file:
            image_path = os.path.join("static/uploads", uploaded_file.filename)
            uploaded_file.save(image_path)
            results = search_similar_image(image_path)
        else:
            return "No image uploaded", 400

    if results is not None:
        return render_template("results.html", results=results.iterrows())
    return "No results found", 404

if __name__ == "__main__":
    app.run(debug=True)
