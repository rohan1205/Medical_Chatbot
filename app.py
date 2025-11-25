import os
import json
import random
import requests
from flask import Flask, request, jsonify, render_template
from langdetect import detect, DetectorFactory

# make language detection deterministic
DetectorFactory.seed = 0

HF_API_KEY = os.environ.get("HF_API_KEY")
if not HF_API_KEY:
    raise RuntimeError("Please set HF_API_KEY env var with your HuggingFace token")

HF_API_URL = "https://api-inference.huggingface.co/models/{}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# models we'll call on the HF Inference API
# these are remote model names; using the same as your original code where possible
ENG_SWA_MODEL = "Rogendo/en-sw"   # English -> Swahili
SWA_ENG_MODEL = "Rogendo/sw-en"   # Swahili -> English
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"  # zero-shot classification

# load intents
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# candidate labels (tags) for zero-shot classification
CANDIDATE_LABELS = [it["tag"] for it in intents.get("intents", [])]

app = Flask(__name__)
app.static_folder = "static"

def hf_post(model_name, payload):
    """Helper to POST to HF inference API and return JSON (or raise)."""
    url = HF_API_URL.format(model_name)
    r = requests.post(url, headers=HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def translate_text(model_name, text, max_length=128):
    """Translate text by calling a text2text-generation model on HuggingFace."""
    payload = {"inputs": text, "parameters": {"max_length": max_length, "num_beams": 5}}
    out = hf_post(model_name, payload)
    # HF returns different shapes sometimes; handle common ones
    if isinstance(out, dict) and out.get("error"):
        raise RuntimeError(f"HF error: {out['error']}")
    # out is usually a list of dicts: [{'generated_text': '...'}]
    if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
        return out[0]["generated_text"]
    # some translation models may return text directly
    if isinstance(out, str):
        return out
    # fallback
    return str(out)

def classify_zero_shot(text, candidate_labels=CANDIDATE_LABELS):
    """Zero-shot classification via HF inference API. Returns top label and score."""
    payload = {"inputs": text, "parameters": {"candidate_labels": candidate_labels}}
    out = hf_post(ZERO_SHOT_MODEL, payload)
    # expected shape: {"labels": [...], "scores": [...]}
    labels = out.get("labels", [])
    scores = out.get("scores", [])
    if labels and scores:
        return labels[0], float(scores[0])
    # fallback: return first label
    return candidate_labels[0], 0.0

def get_response_for_tag(tag):
    """Return a random response text for an intent tag."""
    for it in intents.get("intents", []):
        if it.get("tag") == tag:
            responses = it.get("responses", [])
            if responses:
                return random.choice(responses)
    return "Sorry, I didn't understand that."

@app.route("/")
def home():
    # If you have index.html in templates, render; otherwise a small message
    try:
        return render_template("index.html")
    except Exception:
        return "Vercel Flask server is running. Use /get?msg=your_message"

@app.route("/get")
def get_bot_response():
    userText = request.args.get("msg", "").strip()
    if not userText:
        return "Please provide ?msg=..."

    # 1) detect language locally
    try:
        detected_language = detect(userText)
    except Exception:
        detected_language = "en"  # fallback

    # normalize language codes (langdetect uses ISO-639-1: 'en', 'sw', etc.)
    # We expect 'sw' for Swahili
    is_swahili = detected_language == "sw"

    # 2) if swahili -> translate to english using HF
    if is_swahili:
        try:
            text_for_classification = translate_text(SWA_ENG_MODEL, userText)
        except Exception as e:
            # if translation fails, fallback to original text
            print("Translation SW->EN failed:", e)
            text_for_classification = userText
    else:
        text_for_classification = userText

    # 3) classify using zero-shot (using intent tags as candidate labels)
    try:
        predicted_tag, score = classify_zero_shot(text_for_classification)
    except Exception as e:
        print("Classification failed:", e)
        # fallback: no intent matched
        predicted_tag, score = None, 0.0

    # 4) pick a response from intents.json
    if predicted_tag:
        bot_response_en = get_response_for_tag(predicted_tag)
    else:
        bot_response_en = "Sorry, I didn't understand that."

    # 5) if user originally used swahili, translate back
    if is_swahili:
        try:
            bot_response = translate_text(ENG_SWA_MODEL, bot_response_en)
        except Exception as e:
            print("Translation EN->SW failed:", e)
            bot_response = bot_response_en
    else:
        bot_response = bot_response_en

    return bot_response

if __name__ == "__main__":
    # port and host for local testing
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

