from flask import Flask, render_template, request, jsonify
import fitz
import torch
import requests
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    AutoTokenizer, AutoModelForSequenceClassification
)
from bs4 import BeautifulSoup
import os

app = Flask(__name__)

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text")
    return text.strip()


model_name = "t5-base"
t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
t5_model = t5_model.to(device)


def summarize_text(text, max_input_len=512, max_output_len=150):
    input_text = "summarize: " + text.strip().replace("\n", " ")

    inputs = t5_tokenizer.encode(
        input_text, return_tensors="pt",
        max_length=max_input_len, truncation=True
    ).to(device)

    summary_ids = t5_model.generate(
        inputs,
        max_length=max_output_len,
        min_length=50,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)



GOOGLE_API_KEY = "AIzaSyB-80vdEJEAY5NG_y7GdiUGNQ01B6aA6pg"
CSE_ID = "07bf28c8b8aa44ed7"

def search_similar_cases(query):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": CSE_ID,
        "q": query + " legal judgment case"
    }
    response = requests.get(url, params=params).json()

    results = []
    if "items" in response:
        for item in response["items"][:5]:
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            })
    return results


MODEL_PATH = "models/legalbert_echr_model"

bert_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
bert_model.to(device)
bert_model.eval()

label_map = {0: "no-violation", 1: "violation"}


def predict_outcome(text):
    encoded = bert_tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**encoded)
        probs = torch.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = float(probs[0][pred].item())

    return label_map[pred], confidence


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    text_input = ""

    if "pdf" in request.files and request.files["pdf"].filename != "":
        text_input = extract_text_from_pdf(request.files["pdf"])
    elif "text" in request.form and request.form["text"].strip():
        text_input = request.form["text"].strip()
    else:
        return jsonify({"error": "Please provide text or a PDF."}), 400

    summary = summarize_text(text_input)

    prediction, confidence = predict_outcome(text_input)

    similar_cases = search_similar_cases(summary)

    return jsonify({
        "summary": summary,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "similar_cases": similar_cases
    })

@app.route("/analyze_url", methods=["POST"])
def analyze_url():
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "URL not provided"}), 400

    try:
        page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        page.raise_for_status()
        html = page.text

        soup = BeautifulSoup(html, "html.parser")
        text_input = soup.get_text(separator=" ", strip=True)

        summary = summarize_text(text_input)

        prediction, confidence = predict_outcome(text_input)

        return jsonify({
            "summary": summary,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "source_url": url
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process URL: {str(e)}"}), 500


if __name__ == "__main__":
    print("Starting Legal Summarizer + Violation Predictor App...")
    app.run(debug=True, port=8000)
