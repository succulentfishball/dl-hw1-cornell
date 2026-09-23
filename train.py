import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from d2l import torch as d2l
from torchvision import transforms

from model import LeNet, SETTINGS


SEED = 42
EPOCHS = 100
BATCH_SIZE = 128
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", DEVICE)


MODEL_CONFIG = {
    "dropout": 0.3,
    "weight_decay": 1e-4,
    "lr": 1e-3,
}
OUTPUT_DIR = Path("results3")


def make_loader(dataset, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )

# 90/10 split
def stratified_split(labels, seed):
    rng = torch.Generator().manual_seed(seed)
    train, val = [], []
    for label in range(10):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(len(indices), generator=rng)]
        count = len(indices) // 10
        val.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    return train, val


def create_model(setting, config):
    torch.manual_seed(SEED)

    model = LeNet(setting=setting, **config)

    # Materialize lazy layers using their default initialization.
    model.eval()
    with torch.no_grad():
        model(torch.zeros(2, 1, 28, 28))

    return model.to(DEVICE)


@torch.no_grad()
def evaluate(model, loader):
    """Measure accuracy and mean cross-entropy without changing the model."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)

        total_loss += nn.functional.cross_entropy(
            logits, labels, reduction="sum"
        ).item()

        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    model.train(was_training)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


def train_one(setting, config, train_set, val_loader):
    model = create_model(setting, config)

    optimizer = model.configure_optimizers()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=config["lr"] / 100,  # 0.00001
    )

    train_loader = make_loader(train_set, shuffle=True)
    train_eval_loader = make_loader(train_set)

    history = []
    states = []

    for epoch in range(1, EPOCHS+1):
        current_lr = optimizer.param_groups[0]["lr"]
        model.train()

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            optimizer.step()

        # recompute training accuracy with dropout OFF
        train_metrics = evaluate(model, train_eval_loader)
        val_metrics = evaluate(model, val_loader)

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": current_lr,
        })

        # store current state of model 
        states.append({
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        })

        print(
            f"{setting:12s} | epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_metrics['accuracy']:.2%} | "
            f"validation={val_metrics['accuracy']:.2%}",
            flush=True,
        )

        scheduler.step()

    # get index of current best model weights
    best_index = max(
        range(len(history)),
        key=lambda i: (
            history[i]["val_accuracy"],
            -history[i]["val_loss"],
        ),
    )

    return {
        "config": config.copy(),
        "history": history,
        "states": states,
        "best_index": best_index,
    }


def evaluate_test_sets(results, test_loader):
    for setting, result in results.items():
        model = create_model(setting, result["config"])

        for record, state in zip(
            result["history"],
            result["states"],
        ):
            model.load_state_dict(state)
            metrics = evaluate(model, test_loader)

            record["test_loss"] = metrics["loss"]
            record["test_accuracy"] = metrics["accuracy"]


def print_results(results):
    print("\nValidation-selected checkpoints")
    print(
        f"{'Setting':<15} {'Epoch':>6} "
        f"{'Train':>10} {'Validation':>12} {'Test':>10}"
    )
    print("-" * 57)

    for setting, result in results.items():
        record = result["history"][result["best_index"]]

        print(
            f"{setting:<15} "
            f"{record['epoch']:>6} "
            f"{record['train_accuracy']:>10.2%} "
            f"{record['val_accuracy']:>12.2%} "
            f"{record['test_accuracy']:>10.2%}"
        )


def plot_results(results):
    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 8),
        sharex=True,
        sharey=True,
    )

    for ax, (setting, result) in zip(axes.flat, results.items()):
        history = result["history"]

        epochs = [r["epoch"] for r in history]
        train_acc = [100 * r["train_accuracy"] for r in history]
        test_acc = [100 * r["test_accuracy"] for r in history]

        selected_epoch = history[result["best_index"]]["epoch"]

        ax.plot(epochs, train_acc, label="Train (dropout off)")
        ax.plot(epochs, test_acc, label="Test")

        ax.axvline(
            selected_epoch,
            linestyle=":",
            color="grey",
            label="Validation-selected epoch",
        )

        ax.set_title(setting.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("LeNet on FashionMNIST")
    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / "convergence.png", dpi=200)
    plt.show()


def save_results(results):
    measurements = {}

    for setting, result in results.items():
        best_index = result["best_index"]

        measurements[setting] = {
            "model_config": result["config"],
            "history": result["history"],
            "best_index": best_index,
        }

        torch.save(
            {
                "setting": setting,
                "model_config": result["config"],
                "epoch": result["history"][best_index]["epoch"],
                "model_state": result["states"][best_index],
            },
            OUTPUT_DIR / f"{setting}_best.pt",
        )

    with (OUTPUT_DIR / "history.json").open("w") as file:
        json.dump(
            {
                "seed": SEED,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "results": measurements,
            },
            file,
            indent=2,
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Using d2l's wrapper package
    data = d2l.FashionMNIST(batch_size=BATCH_SIZE)

    full_train_set = data.get_dataloader(True).dataset
    test_set = data.get_dataloader(False).dataset

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    full_train_set.transform = transform
    test_set.transform = transform
    
    train_indices, val_indices = stratified_split(torch.as_tensor(full_train_set.targets), SEED)

    train_set = Subset(full_train_set, train_indices)
    val_set = Subset(full_train_set, val_indices)

    val_loader = make_loader(val_set)
    test_loader = make_loader(test_set)

    with (OUTPUT_DIR / "split.json").open("w") as file:
        json.dump(
            {
                "train": train_set.indices,
                "validation": val_set.indices,
            },
            file,
        )

    results = {}

    for setting in SETTINGS:
        results[setting] = train_one(
            setting,
            MODEL_CONFIG,
            train_set,
            val_loader,
        )

    evaluate_test_sets(results, test_loader)

    save_results(results)
    print_results(results)
    plot_results(results)


if __name__ == "__main__":
    main()