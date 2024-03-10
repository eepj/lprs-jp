import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from typing import Sequence

from models.dataset import LP_CharacterDataset
from models.cnn import LP_CharacterRecognitionCNN


def min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    return (x-x.min()) / (x.max()-x.min())


def stratify_dataset(csv_path: str,
                     image_dir: str,
                     random_state: int) -> tuple[tuple[LP_CharacterDataset], tuple[dict]]:
    # Read csv containing image paths and labels
    df = pd.read_csv(csv_path).dropna()

    s2i = {k: i for i, k in enumerate(sorted(df["label"].unique().tolist()))}
    i2s = {v: k for k, v in s2i.items()}

    train = pd.DataFrame()
    val = pd.DataFrame()
    test = pd.DataFrame()

    # Data split ratio 60-20-20
    train["path"], val["path"], train["label"], val["label"] = train_test_split(df["path"],
                                                                                df["label"],
                                                                                test_size=0.4,
                                                                                random_state=random_state,
                                                                                stratify=df["label"])

    val["path"], test["path"], val["label"], test["label"] = train_test_split(val["path"],
                                                                              val["label"],
                                                                              test_size=0.5,
                                                                              random_state=random_state,
                                                                              stratify=val["label"])

    train = train.dropna().reset_index(drop=True)
    val = val.dropna().reset_index(drop=True)
    test = test.dropna().reset_index(drop=True)

    train = LP_CharacterDataset(image_dir, train["path"], train["label"], s2i)
    val = LP_CharacterDataset(image_dir, val["path"], val["label"], s2i)
    test = LP_CharacterDataset(image_dir, test["path"], test["label"], s2i)

    return (train, val, test), (s2i, i2s)


def visualize_tensors(images: Sequence[torch.Tensor]):
    l = len(images)
    plt.subplots(1, l)
    for i, img in enumerate(images):
        plt.subplot(1, l, i + 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.axis('off')
    plt.show()


def visualize_history(model: LP_CharacterRecognitionCNN,
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


def evaluate_model(model: LP_CharacterRecognitionCNN,
                   dataset: LP_CharacterDataset) -> dict[str: torch.Tensor]:
    dataloader = DataLoader(dataset, 1, shuffle=False)
    res = {}
    res["probs"], res["labels"], res["images"] = model.predict(
        dataloader)
    res["preds"] = torch.argmax(res["probs"], dim=1)
    res["misclf"] = torch.where(res["preds"] != res["labels"])[0]
    res["clf"] = torch.where(res["preds"] == res["labels"])[0]

    for key, value in res.items():
        res[key] = value.cpu()

    accuracy = accuracy_score(res['labels'], res['preds'])
    weighted_f1 = f1_score(res['labels'], res['preds'], average='weighted')
    print(
        f"Evaluated {len(dataset)} isntances: accuracy: {accuracy:.5f} weighted_f1: {weighted_f1:.5f}")

    return res


def visualize_result(result: dict,
                     label_map: dict,
                     title: str | None = None):

    indices = result["misclf"]
    probs = result["probs"]
    labels = result["labels"]
    images = result["images"]

    ncol = 5
    nrows = (len(indices) // ncol) + 1

    _, axs = plt.subplots(nrows, ncol, figsize=(20, 4*nrows))

    for i, z in enumerate(zip(probs[indices], labels[indices], images[indices])):
        prob, label, image = z
        p = torch.max(prob)
        pred_label = torch.argmax(prob).cpu().item()
        label = label.cpu().item()
        ax = plt.subplot(nrows, ncol, i + 1)
        ax.imshow(min_max_normalize(image).cpu().permute(1, 2, 0))
        ax.set_title(
            f"{label_map[pred_label]}@{p:.5f} ({label_map[label]})")
        ax.axis('off')

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
