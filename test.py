import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from d2l import torch as d2l

from model import LeNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model = LeNet(
        setting=checkpoint["setting"],
        **checkpoint["model_config"],
    )

    model.eval()
    with torch.no_grad():
        model(torch.zeros(2, 1, 28, 28))

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    data = d2l.FashionMNIST(batch_size=128)
    test_set = data.get_dataloader(False).dataset

    test_set.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    test_loader = DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            total_loss += nn.functional.cross_entropy(
                logits, labels, reduction="sum"
            ).item()

            correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            total += labels.size(0)

    print(f"Device: {device}")
    print(f"Setting: {checkpoint['setting']}")
    print(f"Selected epoch: {checkpoint['epoch']}")
    print(f"Configuration: {checkpoint['model_config']}")
    print(f"Test loss: {total_loss / total:.4f}")
    print(f"Test accuracy: {correct / total:.2%}")


if __name__ == "__main__":
    main()