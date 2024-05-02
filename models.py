import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms

from io import StringIO
from tqdm import tqdm
from typing import Sequence

"""
Models adapted from:
H. Chen, Y. Lin, and T. Zhao,
'Chinese License Plate Recognition System Based on Convolutional Neural Network',
Highlights in Science, Engineering and Technology, vol. 34, pp. 95-102, 2023.
"""


class CharacterRecognitionConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 device: str = "cpu"):

        super(CharacterRecognitionConvBlock, self).__init__()

        self.device = device

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same")
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x


class CharacterRecognitionCNN(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: Sequence[int],
            layer_sizes: Sequence[int],
            device: str = "cpu",
            transforms: transforms.Transform | None = None):

        super(CharacterRecognitionCNN, self).__init__()

        self.device = device

        d = len(layer_sizes)
        h, w = img_size
        layer_sizes.insert(0, in_channels)
        layer_sizes.append((h // (1 << d)) * (w // (1 << d)) * layer_sizes[-1])

        self.conv = nn.ModuleList()

        for c in range(d):
            self.conv.append(CharacterRecognitionConvBlock(layer_sizes[c], layer_sizes[c + 1], device=self.device))

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(layer_sizes[-1], out_channels)

        self.optimizer = torch.optim.Adam(self.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 30, gamma=0.1)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        self.to(self.device)

        self.transforms = transforms

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for conv in self.conv:
            x = conv(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def train_epoch(self, dataloader: DataLoader) -> tuple[float, float]:

        self.train()

        losses = 0.0
        total_correct = 0
        total_samples = 0

        for batch in dataloader:
            imgs, labels = batch
            if self.transforms is not None:
                imgs = self.transforms(imgs)
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self(imgs.float())

            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            losses += loss

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            accuracy = total_correct / total_samples

        return losses.cpu().item(), accuracy

    def validate_epoch(self, dataloader: DataLoader) -> tuple[float, float]:

        self.eval()

        losses = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                imgs, labels = batch
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                outputs = self(imgs.float())

                loss = self.loss_fn(outputs, labels)
                losses += loss

                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                accuracy = total_correct / total_samples

        return losses.cpu().item(), accuracy

    def train_loop(self,
                   train_dataloader: DataLoader,
                   val_dataloader: DataLoader | None = None,
                   epoch: int = 100,
                   initial_lr: float | None = None) -> None:

        print(f"Training on device: {self.device}")

        if initial_lr is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = initial_lr

        tqdm_io = StringIO()
        tqdm_epoch = tqdm(range(epoch), file=tqdm_io, colour="GREEN", leave=True, ascii=" ░▒██")
        # tqdm update once before entering loop
        print(tqdm_io.getvalue(), end="\n\033[A\r")

        try:
            for e in tqdm_epoch:
                train_acc_loss, train_accuracy = self.train_epoch(
                    train_dataloader)
                train_avg_loss = train_acc_loss / train_dataloader.batch_size
                self.history['train_loss'].append(train_avg_loss)
                self.history['train_accuracy'].append(train_accuracy)

                if val_dataloader is not None:
                    val_acc_loss, val_accuracy = self.validate_epoch(
                        val_dataloader)
                    val_avg_loss = val_acc_loss / val_dataloader.batch_size
                    self.history['val_loss'].append(val_avg_loss)
                    self.history['val_accuracy'].append(val_accuracy)
                    desc = ('train_accuracy', 'val_accuracy')
                else:
                    self.history['val_loss'].append(0.0)
                    self.history['val_accuracy'].append(0.0)
                    desc = ('train_accuracy', 'train_loss')

                self.scheduler.step()

                # Print metrics
                m0 = desc[0], self.history[desc[0]][-1]
                m1 = desc[1], self.history[desc[1]][-1]
                lr_epoch = self.optimizer.param_groups[0]['lr']

                # Work around for updating two lines
                print(tqdm_io.getvalue(), end="\n")
                print(
                    f"Epoch {e+1}: {m0[0]}: {m0[1]:5f}, {m1[0]}: {m1[1]:5f}, lr: {lr_epoch}", end="\033[A\r")

            # tqdm to update once more after loop ends
            print(tqdm_io.getvalue(), end="\n")
            print(
                f"Trained {e+1} epochs: {m0[0]}: {m0[1]:5f}, {m1[0]}: {m1[1]:5f}, lr: {lr_epoch}", end="\033[A\r\n\n")

        except KeyboardInterrupt:
            pass

    def predict(self,
                dataset: Dataset) -> tuple[torch.Tensor]:

        self.eval()

        probs = torch.Tensor([]).to(self.device)
        labels = torch.Tensor([]).to(self.device)
        imgs = torch.Tensor([]).to(self.device)

        print(f"Evaluating on device: {self.device}")

        with torch.no_grad():
            tqdm_io = StringIO()
            tqdm_dataset = tqdm(dataset, file=tqdm_io, colour="GREEN", leave=True, ascii=" ░▒██")

            for sample in tqdm_dataset:
                img, label = sample
                img = img.to(self.device)
                label = label.to(self.device)

                output = self(img.float())
                output = torch.softmax(output, dim=1)

                probs = torch.cat((probs, output))
                labels = torch.cat((labels, torch.Tensor([label]).to(self.device)))
                imgs = torch.cat([imgs, img])

                # For consistency with `train_epoch` function
                print(tqdm_io.getvalue(), end="\r")

            # tqdm to update once more after loop ends
            print(tqdm_io.getvalue(), end="\r")

        return probs, labels, imgs

    def clear_history(self):

        for key in self.history.keys():
            self.history[key].clear()
