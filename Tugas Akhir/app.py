import pickle
import faiss
from PIL import Image
import torch
import faiss
import clip
import torch
from flask import Flask, render_template, request
import os


# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print("CLIP test successful!")


#Setup Flask
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)



# Load preprocessed data
with open("preprocessed/embeddings.pkl", "rb") as f:
    styles_df = pickle.load(f)

print("Loading text index...")
text_index = faiss.read_index("preprocessed/text_index.faiss")
print("Text index loaded.")
print("Loading image index...")
image_index = faiss.read_index("preprocessed/image_index.faiss")
print("Image index loaded.")

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

def search_similar_text(query, k=5):
    query_embedding = get_clip_text_embeddings([query])
    distances, indices = text_index.search(query_embedding, k)
    return styles_df.iloc[indices[0]]

def search_similar_image(image_path, k=5):
    query_embedding = get_clip_image_embeddings(image_path)
    distances, indices = image_index.search(query_embedding, k)
    return styles_df.iloc[indices[0]]

# Routes
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
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
            uploaded_file.save(image_path)
            results = search_similar_image(image_path)
        else:
            return "No image uploaded", 400

    if results is not None:
        return render_template("results.html", results=results.iterrows())
    return "No results found", 404

if __name__ == "__main__":
    app.run(debug=True)
