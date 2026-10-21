# test_audio_accuracy.py
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import numpy as np
import pandas as pd
from torchvision import models as tv_models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import os

os.makedirs("outputs/results", exist_ok=True)

class ResNetSpoof(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = tv_models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc    = nn.Linear(512, 1)
    def forward(self, x):
        return self.model(x).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ResNetSpoof()
model.load_state_dict(torch.load("models/audio/spoof_resnet18.pth", map_location=device))
model.to(device)
model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=1024, hop_length=512, n_mels=80
)

def predict_audio(path):
    try:
        waveform, sr = sf.read(str(path))
        waveform     = torch.tensor(waveform).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.mean(dim=1, keepdim=True).T
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        max_len = 16000 * 4
        if waveform.shape[1] > max_len:
            waveform = waveform[:, :max_len]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, max_len - waveform.shape[1]))
        mel = mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        mel = mel.unsqueeze(0).to(device)
        with torch.no_grad():
            risk = torch.sigmoid(model(mel)).item()
        return risk
    except:
        return None

# Load test CSV — uses your train.csv label structure
# label 1 = spoof/fake, label 0 = bonafide/real
csv_path = "train.csv"
if not Path(csv_path).exists():
    print("train.csv not found — searching for audio files directly...")
    # Fall back: grab files from raw audio folder
    audio_dir = Path("data/audio/raw")
    all_files = list(audio_dir.rglob("*.flac"))[:100]
    print(f"Found {len(all_files)} flac files — no labels available for accuracy test")
    print("Please provide train.csv with columns: path, label")
    exit()

df      = pd.read_csv(csv_path).dropna(subset=['path','label'])
# Test on first 200 samples (100 real + 100 fake if balanced)
real_df = df[df['label'] == 0].head(100)
fake_df = df[df['label'] == 1].head(100)
test_df = pd.concat([real_df, fake_df], ignore_index=True)

print(f"Testing on {len(test_df)} audio files...")

y_true, y_pred = [], []
for i, row in test_df.iterrows():
    risk = predict_audio(row['path'])
    if risk is None:
        continue
    pred  = 1 if risk >= 0.5 else 0
    y_true.append(int(row['label']))
    y_pred.append(pred)
    if (len(y_true)) % 20 == 0:
        print(f"  Processed {len(y_true)}/{len(test_df)}...")

acc    = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["Bonafide/Real", "Spoof/Fake"])
cm     = confusion_matrix(y_true, y_pred)

lines = [
    "=" * 60,
    "  NeuraShield — Audio Model Accuracy Report",
    "=" * 60,
    f"  Model     : ResNet18 on Mel Spectrogram",
    f"  Dataset   : ASVspoof 2019",
    f"  Tested on : {len(y_true)} audio files",
    f"  Accuracy  : {acc*100:.2f}%",
    "",
    "Classification Report:",
    report,
    "Confusion Matrix:",
    f"  TN={cm[0][0]}  FP={cm[0][1]}",
    f"  FN={cm[1][0]}  TP={cm[1][1]}",
    "=" * 60
]

output = "\n".join(lines)
print(output)

with open("outputs/results/audio_model_accuracy.txt", "w") as f:
    f.write(output)

print("\nSaved to: outputs/results/audio_model_accuracy.txt") 