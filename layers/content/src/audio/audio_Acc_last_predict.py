import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from pathlib import Path
from torchvision import models as tv_models

MODEL_PATH = "models/audio/spoof_resnet18.pth"
REAL_DIR   = Path("data/audio/test/real")
FAKE_DIR   = Path("data/audio/test/fake")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class ResNetSpoof(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = tv_models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)

model = ResNetSpoof()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✓ Model loaded\n")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=1024, hop_length=512, n_mels=80
)

def get_risk_score(file_path: str):
    try:
        waveform, sr = sf.read(file_path)
        waveform = torch.tensor(waveform).float()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        max_len = 16000 * 4
        if waveform.shape[1] > max_len:
            waveform = waveform[:, :max_len]
        else:
            pad = max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        mel = mel_transform(waveform)
        mel = torch.log(mel + 1e-6)
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        mel = mel.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logit = model(mel)
            risk = torch.sigmoid(logit).item()
        return risk
    except Exception as e:
        print(f"Error on {file_path}: {e}")
        return None

# Collect scores
real_scores = []
fake_scores = []

print("Collecting scores from REAL files...")
for f in list(REAL_DIR.glob("*"))[:15]:
    score = get_risk_score(str(f))
    if score is not None:
        real_scores.append(score)

print("Collecting scores from FAKE files...")
for f in list(FAKE_DIR.glob("*"))[:15]:
    score = get_risk_score(str(f))
    if score is not None:
        fake_scores.append(score)

print("\n" + "="*60)
print("SCORE DISTRIBUTION")
print("="*60)
print(f"Real scores  → min={min(real_scores):.4f}  max={max(real_scores):.4f}  avg={sum(real_scores)/len(real_scores):.4f}")
print(f"Fake scores  → min={min(fake_scores):.4f}  max={max(fake_scores):.4f}  avg={sum(fake_scores)/len(fake_scores):.4f}")

print("\n" + "="*60)
print("THRESHOLD EXPERIMENT")
print("="*60)

for thresh in [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]:
    real_correct = sum(1 for s in real_scores if s < thresh)
    fake_correct = sum(1 for s in fake_scores if s >= thresh)
    
    real_acc = real_correct / len(real_scores) * 100
    fake_acc = fake_correct / len(fake_scores) * 100
    overall  = (real_correct + fake_correct) / (len(real_scores) + len(fake_scores)) * 100
    
    print(f"Threshold {thresh:.2f} → Real: {real_acc:.1f}%  |  Fake: {fake_acc:.1f}%  |  Overall: {overall:.1f}%")