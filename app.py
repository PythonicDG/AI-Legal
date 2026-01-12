from flask import Flask, render_template, request, jsonify
import fitz
import torch
import requests
import time
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    AutoTokenizer, AutoModelForSequenceClassification
)
from bs4 import BeautifulSoup


app = Flask(__name__)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_CHARS = 8000
SUMMARY_PRED_CHARS = 2000


def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text")
    return text.strip()[:MAX_CHARS]


t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

if device == "cuda":
    t5_model = t5_model.half()

def summarize_text(text):
    input_text = (
        "summarize in a detailed, comprehensive manner, "
        "covering all key points and explanations: "
        + text.replace("\n", " ")
    )

    inputs = t5_tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    summary_ids = t5_model.generate(
        inputs,
        max_length=500,
        min_length=150,
        num_beams=2,
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

    response = requests.get(url, params=params, timeout=10).json()

    return [
        {
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet")
        }
        for item in response.get("items", [])[:5]
    ]


MODEL_PATH = "models/legalbert_echr_model"

bert_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
bert_model.eval()

if device == "cuda":
    bert_model = bert_model.half()

label_map = {0: "no-violation", 1: "violation"}

def predict_outcome(text):
    encoded = bert_tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    outputs = bert_model(**encoded)
    probs = torch.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = float(probs[0][pred].item()) * 100

    return label_map[pred], round(confidence, 2)


ISSUE_KEYWORDS = [
    "contract", "breach", "liability", "damages",
    "negligence", "fraud", "jurisdiction",
    "employment", "property", "constitutional"
]

def extract_key_issues(text):
    found = {kw.capitalize() for kw in ISSUE_KEYWORDS if kw in text.lower()}
    return ", ".join(found) if found else "Not clearly identified"

def detect_jurisdiction(text):
    t = text.lower()
    if "supreme court" in t or "u.s.c." in t:
        return "U.S. Federal Court"
    if "high court" in t:
        return "State High Court"
    if "district court" in t:
        return "District Court"
    if "tribunal" in t:
        return "Tribunal"
    return "Jurisdiction not identified"


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    start_time = time.time()

    text_input = ""

    if "pdf" in request.files and request.files["pdf"].filename:
        text_input = extract_text_from_pdf(request.files["pdf"])
    elif "text" in request.form and request.form["text"].strip():
        text_input = request.form["text"].strip()[:MAX_CHARS]
    else:
        return jsonify({"error": "Please provide text or a PDF."}), 400

    summary = summarize_text(text_input)
    prediction, confidence = predict_outcome(summary[:SUMMARY_PRED_CHARS])
    similar_cases = search_similar_cases(summary)

    analysis_time = round(time.time() - start_time, 2)

    return jsonify({
        "summary": summary,
        "prediction": prediction,
        "confidence": confidence,
        "similar_cases": similar_cases,
        "analysisTime": f"{analysis_time} seconds",
        "docLength": f"{len(text_input.split())} words",
        "keyIssues": extract_key_issues(text_input),
        "jurisdiction": detect_jurisdiction(text_input)
    })

@app.route("/analyze_url", methods=["POST"])
def analyze_url():
    data = request.get_json()
    url = data.get("url")

    if not url:
        return jsonify({"error": "URL not provided"}), 400

    start_time = time.time()

    page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    page.raise_for_status()

    soup = BeautifulSoup(page.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text_input = " ".join(soup.get_text().split())[:MAX_CHARS]

    summary = summarize_text(text_input)
    prediction, confidence = predict_outcome(summary[:SUMMARY_PRED_CHARS])

    return jsonify({
        "summary": summary,
        "prediction": prediction,
        "confidence": confidence,
        "analysisTime": f"{round(time.time() - start_time, 2)} seconds",
        "docLength": f"{len(text_input.split())} words",
        "keyIssues": extract_key_issues(text_input),
        "jurisdiction": detect_jurisdiction(text_input),
        "source_url": url
    })


if __name__ == "__main__":
    print("Starting REAL Legal Analysis App...")
    app.run(debug=True, port=8000)
