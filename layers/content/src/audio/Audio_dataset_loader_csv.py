import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

class SpoofDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=80
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["path"]
        label = self.df.iloc[idx]["label"]

        import soundfile as sf

        waveform, sr = sf.read(path)

        waveform = torch.tensor(waveform).float()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.mean(dim=1, keepdim=True).T

        if sr != 16000:
            waveform = torchaudio.functional.resample(
                waveform,
                sr,
                16000
            )

        max_len = 16000 * 4

        if waveform.shape[1] > max_len:
            waveform = waveform[:, :max_len]
        else:
            pad = max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(
                waveform,
                (0, pad)
            )

        mel = self.mel_transform(waveform)

        mel = torch.log(mel + 1e-6)

        mel = (mel - mel.mean()) / (mel.std() + 1e-9)

        return mel, torch.tensor(label).float()


if __name__ == "__main__":

    dataset = SpoofDataset("train.csv")

    print("Dataset size:", len(dataset))

    mel, label = dataset[0]

    print("Mel shape:", mel.shape)
    print("Label:", label)