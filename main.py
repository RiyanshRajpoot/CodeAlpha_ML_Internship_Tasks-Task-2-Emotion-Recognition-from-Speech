import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# RAVDESS filename convention: modality-vocal_channel-emotion-intensity-statement-
# repetition-actor.wav. Emotion code is the third field.
RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

def extract_features(file_path, duration=3.0, sample_rate=22050):
    """Return mean and standard deviation of 40 MFCC coefficients."""
    audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {file_path}")
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])

def label_from_ravdess(file_path):
    parts = Path(file_path).stem.split("-")
    if len(parts) < 3 or parts[2] not in RAVDESS_EMOTIONS:
        raise ValueError(f"Cannot read RAVDESS emotion code from filename: {file_path}")
    return RAVDESS_EMOTIONS[parts[2]]

def load_dataset(data_dir):
    paths = list(Path(data_dir).rglob("*.wav"))
    if not paths:
        raise FileNotFoundError(
            f"No WAV files found under '{data_dir}'. Download/extract RAVDESS audio there."
        )
    features, labels = [], []
    skipped = 0
    for path in paths:
        try:
            features.append(extract_features(path))
            labels.append(label_from_ravdess(path))
        except Exception as exc:
            skipped += 1
            print(f"Skipping {path.name}: {exc}")
    if not features:
        raise RuntimeError("No usable WAV files were found.")
    print(f"Loaded {len(features)} audio files; skipped {skipped}.")
    return np.asarray(features), np.asarray(labels)

def train(data_dir, classifier="random_forest"):
    X, y = load_dataset(data_dir)
    if len(np.unique(y)) < 2:
        raise ValueError("At least two emotion classes are required for training.")
    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    if classifier == "svm":
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, probability=True))
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("\\nClassifier:", classifier)
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("\\nClassification report:\\n", classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix (labels:", list(model.classes_), "):\\n", confusion_matrix(y_test, pred))
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/speech_emotion_model.joblib")
    print("\\nSaved model to models/speech_emotion_model.joblib")

def predict(audio_path, model_path):
    model = joblib.load(model_path)
    features = extract_features(audio_path).reshape(1, -1)
    print("Predicted emotion:", model.predict(features)[0])

def main():
    parser = argparse.ArgumentParser(description="CodeAlpha Task 2: Speech Emotion Recognition")
    parser.add_argument("--data", default="data/ravdess", help="Folder containing extracted RAVDESS WAV files")
    parser.add_argument("--classifier", choices=["random_forest", "svm"], default="random_forest")
    parser.add_argument("--predict", help="Path to one WAV file for prediction")
    parser.add_argument("--model", default="models/speech_emotion_model.joblib")
    args = parser.parse_args()
    if args.predict:
        predict(args.predict, args.model)
    else:
        train(args.data, args.classifier)

if __name__ == "__main__":
    main()