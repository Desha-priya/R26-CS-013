if __name__ == "__main__":

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from Audio_dataset_loader_csv import SpoofDataset
    from model import ResNetSpoof

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", DEVICE)

    dataset = SpoofDataset("train.csv")
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )

    model = ResNetSpoof().to(DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 10

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        loop = tqdm(loader)

        for mel, labels in loop:
            mel = mel.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(mel)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = 100 * correct / total
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "spoof_resnet18.pth")
    print("Model saved.")