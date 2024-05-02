import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from typing import Sequence

from dataset import CharacterDataset
from models import CharacterRecognitionCNN


def min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    return (x-x.min()) / (x.max()-x.min())


def stratify_dataset(csv_path: str,
                     img_dir: str,
                     random_state: int) -> dict:

    # Read csv containing image paths and labels
    df = pd.read_csv(csv_path).dropna()

    stoi = {k: i for i, k in enumerate(sorted(df["label"].unique().tolist()))}
    itos = {v: k for k, v in stoi.items()}

    train_imgs = pd.DataFrame()
    val_imgs = pd.DataFrame()
    test_imgs = pd.DataFrame()

    # Data split ratio 60-20-20
    temp = train_test_split(df["path"], df["label"], test_size=0.4, random_state=random_state, stratify=df["label"])
    train_imgs["path"], temp_path, train_imgs["label"], temp_label = temp

    temp = train_test_split(temp_path, temp_label, test_size=0.5, random_state=random_state, stratify=temp_label)
    val_imgs["path"], test_imgs["path"], val_imgs["label"], test_imgs["label"] = temp

    train_imgs = train_imgs.dropna().reset_index(drop=True)
    val_imgs = val_imgs.dropna().reset_index(drop=True)
    test_imgs = test_imgs.dropna().reset_index(drop=True)

    train_set = CharacterDataset(img_dir, train_imgs["path"], train_imgs["label"], stoi)
    val_set = CharacterDataset(img_dir, val_imgs["path"], val_imgs["label"], stoi)
    test_set = CharacterDataset(img_dir, test_imgs["path"], test_imgs["label"], stoi)

    splits = {
        "split": {"train": train_set, "val": val_set, "test": test_set},
        "map": {"stoi": stoi, "itos": itos}
    }

    return splits


def visualize_tensors(imgs: Sequence[torch.Tensor]):

    l = len(imgs)
    plt.subplots(1, l)
    for i, img in enumerate(imgs):
        plt.subplot(1, l, i + 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.axis("off")
    plt.show()


def visualize_history(model: CharacterRecognitionCNN,
                      title: str = ""):
    history = model.history

    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.twinx()
    plt.plot(history["train_accuracy"],
             label="Train Accuracy", color="green")
    plt.plot(history["val_accuracy"],
             label="Validation Accuracy", color="orange")

    plt.ylabel("Accuracy")
    plt.legend()

    plt.title(f"Training and Validation Loss {title}")

    plt.show()


def evaluate_model(model: CharacterRecognitionCNN,
                   dataset: CharacterDataset) -> dict[str: torch.Tensor]:

    dataloader = DataLoader(dataset, 1, shuffle=False)
    results = {}
    results["probs"], results["labels"], results["images"] = model.predict(dataloader)
    results["preds"] = torch.argmax(results["probs"], dim=1)
    results["misclf"] = torch.where(results["preds"] != results["labels"])[0]
    results["clf"] = torch.where(results["preds"] == results["labels"])[0]

    for key, value in results.items():
        results[key] = value.cpu()

    accuracy = accuracy_score(results["labels"], results["preds"])
    weighted_f1 = f1_score(results["labels"], results["preds"], average="weighted")
    print(
        f"Evaluated {len(dataset)} isntances: accuracy: {accuracy:.5f} weighted_f1: {weighted_f1:.5f}")

    return results


def visualize_result(result: dict,
                     label_map: dict,
                     title: str | None = None):

    indices = result["misclf"]
    probs = result["probs"]
    labels = result["labels"]
    imgs = result["images"]

    ncol = 5
    nrows = (len(indices) // ncol) + 1

    _, axs = plt.subplots(nrows, ncol, figsize=(20, 4*nrows))

    for i, z in enumerate(zip(probs[indices], labels[indices], imgs[indices])):
        prob, label, img = z
        p = torch.max(prob)
        pred_label = torch.argmax(prob).cpu().item()
        label = label.cpu().item()
        ax = plt.subplot(nrows, ncol, i + 1)
        ax.imshow(min_max_normalize(img).cpu().permute(1, 2, 0))
        ax.set_title(
            f"{label_map[pred_label]}@{p:.5f} ({label_map[label]})")
        ax.axis("off")

    if len(axs.shape) > 1:
        ax_last = axs[-1]
    else:
        ax_last = axs

    for ax in ax_last[len(indices) % ncol:]:
        ax.remove()

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()
