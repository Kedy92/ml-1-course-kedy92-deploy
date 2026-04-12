from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .metrics_utils import classification_metrics


class CreditDefaultDataset(Dataset):
    def __init__(self, categorical_data: np.ndarray, numerical_data: np.ndarray, labels: np.ndarray):
        self.categorical_data = torch.tensor(categorical_data, dtype=torch.long)
        self.numerical_data = torch.tensor(numerical_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.categorical_data[idx], self.numerical_data[idx], self.labels[idx]


class TabularMLP(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        num_numerical_features: int,
        hidden_dims: list[int],
        dropout: float,
    ):
        super().__init__()
        embedding_dims = [min(32, (cardinality + 1) // 2) for cardinality in cardinalities]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim) for cardinality, embedding_dim in zip(cardinalities, embedding_dims)]
        )
        total_input_dim = sum(embedding_dims) + num_numerical_features

        layers: list[nn.Module] = []
        in_features = total_input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, categorical_inputs: torch.Tensor, numerical_inputs: torch.Tensor) -> torch.Tensor:
        embedded = [embedding(categorical_inputs[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        features = torch.cat(embedded + [numerical_inputs], dim=1)
        return self.network(features).squeeze(1)


@dataclass
class MLPTrainingResult:
    model: TabularMLP
    config: dict
    history: list[dict]
    val_metrics: dict
    test_metrics: dict
    test_probabilities: np.ndarray


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate_model(
    model: TabularMLP,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for cats, nums, labels in loader:
            cats = cats.to(device)
            nums = nums.to(device)
            labels = labels.to(device)
            logits = model(cats, nums)
            loss = loss_fn(logits, labels)
            probs = torch.sigmoid(logits)
            losses.append(loss.item() * len(labels))
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    total = len(loader.dataset)
    return (
        float(np.sum(losses) / total),
        np.concatenate(all_probs),
        np.concatenate(all_targets),
    )


def train_mlp_configs(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    cardinalities: list[int],
    num_numerical_features: int,
    pos_weight: float,
    device: torch.device,
    configs: list[dict],
) -> tuple[list[dict], MLPTrainingResult]:
    summaries = []
    best_result: MLPTrainingResult | None = None

    for config in configs:
        model = TabularMLP(
            cardinalities=cardinalities,
            num_numerical_features=num_numerical_features,
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        train_loader = build_dataloader(train_dataset, config["batch_size"], shuffle=True)
        val_loader = build_dataloader(val_dataset, config["batch_size"], shuffle=False)
        test_loader = build_dataloader(test_dataset, config["batch_size"], shuffle=False)

        best_val_auc = -np.inf
        best_state = None
        history = []
        patience_left = config["patience"]

        for epoch in range(1, config["epochs"] + 1):
            model.train()
            train_losses = []
            for cats, nums, labels in train_loader:
                cats = cats.to(device)
                nums = nums.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(cats, nums)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item() * len(labels))

            train_loss = float(np.sum(train_losses) / len(train_loader.dataset))
            val_loss, val_probs, val_targets = evaluate_model(model, val_loader, loss_fn, device)
            val_metrics = classification_metrics(val_targets, val_probs)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auc": val_metrics["auc_roc"],
                    "val_f1": val_metrics["f1"],
                }
            )

            if val_metrics["auc_roc"] > best_val_auc:
                best_val_auc = val_metrics["auc_roc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = config["patience"]
            else:
                patience_left -= 1
                if patience_left == 0:
                    break

        assert best_state is not None
        model.load_state_dict(best_state)

        val_loss, val_probs, val_targets = evaluate_model(model, val_loader, loss_fn, device)
        test_loss, test_probs, test_targets = evaluate_model(model, test_loader, loss_fn, device)
        final_val_metrics = classification_metrics(val_targets, val_probs)
        final_val_metrics["loss"] = val_loss
        final_test_metrics = classification_metrics(test_targets, test_probs)
        final_test_metrics["loss"] = test_loss

        summaries.append(
            {
                "config_name": config["name"],
                "hidden_dims": config["hidden_dims"],
                "dropout": config["dropout"],
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "val_auc": final_val_metrics["auc_roc"],
                "val_f1": final_val_metrics["f1"],
                "test_auc": final_test_metrics["auc_roc"],
                "test_f1": final_test_metrics["f1"],
            }
        )

        if best_result is None or final_val_metrics["auc_roc"] > best_result.val_metrics["auc_roc"]:
            best_result = MLPTrainingResult(
                model=model,
                config=config,
                history=history,
                val_metrics=final_val_metrics,
                test_metrics=final_test_metrics,
                test_probabilities=test_probs,
            )

    assert best_result is not None
    return summaries, best_result
