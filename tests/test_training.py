import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.train_model import run_epoch


class TrainingTests(unittest.TestCase):
    def test_run_epoch_returns_metrics(self) -> None:
        model = nn.Linear(4, 2)
        dataset = TensorDataset(torch.randn(6, 4), torch.tensor([0, 1, 0, 1, 0, 1]))
        loader = DataLoader(dataset, batch_size=2)
        loss, accuracy = run_epoch(
            model,
            loader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
        )
        self.assertGreaterEqual(loss, 0.0)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_output_directory_can_be_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested"
            output.mkdir(parents=True)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
