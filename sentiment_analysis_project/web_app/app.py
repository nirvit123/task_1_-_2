from flask import Flask, render_template, request
import torch
import pickle
from transformers import BertTokenizer, BertModel
import numpy as np

app = Flask(__name__)

# Load models
device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained("bert_tokenizer")
bert_model = BertModel.from_pretrained("bert_model")
bert_model.to(device)
bert_model.eval()

with open("sentiment_lr_bert.pkl", "rb") as f:
    clf = pickle.load(f)

def get_bert_embedding(text):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output = bert_model(**encoded)

    cls_embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embedding

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    if request.method == "POST":
        review = request.form["review"]
        embedding = get_bert_embedding(review)
        prediction = clf.predict(embedding)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
