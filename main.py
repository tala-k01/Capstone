from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import requests
import re
import time
import base64
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from transformers import BertTokenizer, BertForSequenceClassification
from your_utils_module import infer_mbti_from_mood
import openai

load_dotenv()
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__, template_folder="templates")
CORS(app)

MODEL_DIR = "models"
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_mood.pkl"))
mood_encoder = joblib.load(os.path.join(MODEL_DIR, "mood_encoder.pkl"))

bert_model_path = os.path.join(MODEL_DIR, "bert_sentiment_model.pt")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
bert_model.load_state_dict(torch.load(bert_model_path, map_location=torch.device("cpu")))
bert_model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class FusionNet(nn.Module):
    def __init__(self, input_size=14, hidden_size=32, output_size=8):
        super(FusionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

fusion_model = FusionNet()
fusion_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "fusion_model1.pth"), map_location=torch.device("cpu")))
fusion_model.eval()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate_mbti_description', methods=['POST'])
def generate_mbti_description():
    data = request.get_json()
    mbti = data.get("mbti")

    if not mbti:
        return jsonify({"error": "MBTI type is missing"}), 400

    prompt = f"Write a short, creative, and relatable personality description for someone with the MBTI type {mbti}. Use a warm tone, and keep it under 3 sentences."
    sentence = f"Generate a description for the MBTI type {mbti}."
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        description = response.choices[0].message.content.strip()
        return jsonify({"description": description})
    except Exception as e:
        return jsonify({"error": "OpenAI request failed", "details": str(e)}), 500



def get_token():
    auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
    url = "https://accounts.spotify.com/api/token"
    headers = {"Authorization": "Basic " + auth_base64, "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, headers=headers, data=data)
    return response.json()["access_token"]

def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

def get_playlist_tracks_only(playlist_url, token):
    playlist_id = playlist_url.split("/")[-1].split("?")[0]
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = get_auth_header(token)
    response = requests.get(url, headers=headers)
    items = response.json()["items"]
    songs = []
    for item in items:
        track = item["track"]
        if track and track.get("id"):
            songs.append({"artist": track["artists"][0]["name"], "title": track["name"], "id": track["id"]})
    return songs

def normalize(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower())

def scrape_audio_features(artist, title):
    try:
        query = f"{artist} {title}".replace(" ", "%20")
        url = f"https://songdata.io/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        card = soup.find("div", class_="card-body")
        if not card:
            return None
        text = card.get_text(separator="\n")
        features = {}
        if "Danceability" in text:
            features["danceability"] = float(re.findall(r"Danceability: ([\d.]+)", text)[0]) / 100
        if "Energy" in text:
            features["energy"] = float(re.findall(r"Energy: ([\d.]+)", text)[0]) / 100
        if "Valence" in text:
            features["valence"] = float(re.findall(r"Valence: ([\d.]+)", text)[0]) / 100
        if "Tempo" in text:
            features["tempo"] = float(re.findall(r"Tempo: ([\d.]+)", text)[0])
        return features if features else None
    except:
        return None

def get_audio_features_local_or_scrape(songs, local_csv_path="data/spotify_songs.csv"):
    local_df = pd.read_csv(local_csv_path)
    enriched_songs = []
    for song in songs:
        artist = normalize(song["artist"])
        title = normalize(song["title"])
        features = {"artist": song["artist"], "title": song["title"], "danceability": None, "energy": None, "valence": None, "tempo": None}
        match = local_df[
            local_df["track_artist"].apply(lambda x: normalize(str(x))).str.contains(artist) &
            local_df["track_name"].apply(lambda x: normalize(str(x))).str.contains(title)
        ]
        if not match.empty:
            row = match.iloc[0]
            features.update({"danceability": row["danceability"], "energy": row["energy"], "valence": row["valence"], "tempo": row["tempo"]})
        else:
            scraped = scrape_audio_features(song["artist"], song["title"])
            if scraped:
                features.update(scraped)
        enriched_songs.append(features)
        time.sleep(1)
    return pd.DataFrame(enriched_songs)

def get_lyrics(artist, title):
    try:
        url = f"https://api.lyrics.ovh/v1/{artist}/{title}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("lyrics", None)
        return None
    except:
        return None

def predict_lyrics_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits[0]
    return logits.tolist()

@app.route('/predict_playlist_rf', methods=['POST'])
def predict_playlist_rf():
    data = request.json
    playlist_url = data.get("playlist_url")
    if not playlist_url:
        return jsonify({"error": "Missing playlist_url"}), 400

    try:
        token = get_token()
        songs = get_playlist_tracks_only(playlist_url, token)
        audio_df = get_audio_features_local_or_scrape(songs)
        predictions = []
        mood_counts = []
        for _, row in audio_df.iterrows():
            audio_ok = all(pd.notna([row['valence'], row['energy'], row['danceability'], row['tempo']]))
            lyrics = get_lyrics(row["artist"], row["title"])
            sentiment_logits = predict_lyrics_sentiment(lyrics) if lyrics else None
            mood = "Unknown"
            model_used = "none"
            try:
                if audio_ok and sentiment_logits:
                    audio_vec = rf_model.predict_proba(pd.DataFrame([row[['valence', 'energy', 'danceability', 'tempo']]]))[0]
                    fusion_input = np.concatenate([audio_vec, sentiment_logits])
                    if len(fusion_input) < 14:
                        fusion_input = np.pad(fusion_input, (0, 14 - len(fusion_input)), mode='constant')
                    fusion_tensor = torch.tensor(fusion_input, dtype=torch.float32).unsqueeze(0)
                    output = fusion_model(fusion_tensor)
                    mood_idx = torch.argmax(output, dim=1).item()
                    mood = mood_encoder.inverse_transform([mood_idx])[0]
                    model_used = "fusion"
                elif audio_ok:
                    audio_vec = rf_model.predict_proba(pd.DataFrame([row[['valence', 'energy', 'danceability', 'tempo']]]))[0]
                    mood_idx = np.argmax(audio_vec)
                    mood = mood_encoder.inverse_transform([mood_idx])[0]
                    model_used = "audio-only"
                elif sentiment_logits:
                    mood_idx = np.argmax(sentiment_logits)
                    mood = mood_encoder.inverse_transform([mood_idx])[0]
                    model_used = "lyrics-only"
            except Exception as e:
                print(f"Prediction failed for {row['title']} by {row['artist']}: {e}")

            predictions.append({
                "artist": row["artist"],
                "title": row["title"],
                "predicted_mood": mood,
                "model_used": model_used,
                "lyrics": lyrics if lyrics else "Lyrics not found."
            })

            if mood != "Unknown":
                mood_counts.append(mood)

        mood_distribution = {m: mood_counts.count(m)/len(mood_counts) for m in set(mood_counts)}
        mbti = infer_mbti_from_mood(mood_distribution)
        return jsonify({"songs": predictions, "mbti": mbti})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
