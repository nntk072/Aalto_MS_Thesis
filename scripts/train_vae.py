"""Training script for Variational Autoencoder.

This script trains the VAE on historical pre-NY session data (01:05-16:29 UTC+3).
The VAE compresses the entire 15.5-hour sequence into a latent embedding z
that captures the day's narrative (e.g., tight range vs. breakout).

The trained encoder is then frozen and used as a feature extractor for the
RL agent's observation space.

Usage:
    python scripts/train_vae.py [--config config/vae.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

# Add project root to path before importing VAE
sys.path.insert(0, str(Path(__file__).parent.parent))
from quant_rl.models.vae import VAE  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PreNYSequenceDataset(Dataset):
    """Dataset for pre-NY session sequences.

    Each sample is a sequence of 5-minute OHLCV bars from 01:05 to 16:29 UTC+3.
    """

    def __init__(self, data: np.ndarray) -> None:
        """Initialize dataset.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (n_samples, seq_len, n_features) containing
            the pre-NY sequences.
        """
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single sample.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        torch.Tensor
            Sequence tensor of shape (seq_len, n_features).
        """
        return torch.from_numpy(self.data[idx]).float()


class VAELoss:
    """VAE loss function combining reconstruction and KL divergence."""

    def __init__(self, reconstruction_weight: float = 1.0, kl_weight: float = 1.0) -> None:
        """Initialize VAE loss.

        Parameters
        ----------
        reconstruction_weight : float
            Weight for the reconstruction loss (MSE).
        kl_weight : float
            Weight for the KL divergence loss.
        """
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.mse_loss = nn.MSELoss(reduction="sum")

    def __call__(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VAE loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, n_features).
        x_recon : torch.Tensor
            Reconstructed tensor of shape (batch_size, seq_len, n_features).
        mu : torch.Tensor
            Mean of latent distribution of shape (batch_size, latent_dim).
        log_var : torch.Tensor
            Log-variance of latent distribution of shape (batch_size, latent_dim).

        Returns
        -------
        torch.Tensor
            Total loss value.
        """
        # Reconstruction loss (MSE)
        recon_loss = self.mse_loss(x_recon, x)

        # KL divergence loss
        # KL(N(mu, sigma^2), N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss
        total_loss = self.reconstruction_weight * recon_loss + self.kl_weight * kl_loss

        return total_loss


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(
    config: dict,
) -> tuple[PreNYSequenceDataset, PreNYSequenceDataset, PreNYSequenceDataset]:
    """Load and prepare training, validation, and test data.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing data paths and parameters.

    Returns
    -------
    tuple[PreNYSequenceDataset, PreNYSequenceDataset, PreNYSequenceDataset]
        Training, validation, and test datasets.
    """
    data_path = Path(config["data_path"])
    seq_len = config["seq_len"]
    n_features = config["n_features"]

    # Load data from numpy files
    # Expected format: (n_samples, seq_len, n_features)
    data = np.load(data_path)

    # Ensure correct shape
    if data.ndim != 3:
        raise ValueError(f"Expected 3D data (n_samples, seq_len, n_features), got {data.ndim}D")

    if data.shape[1] != seq_len:
        logger.warning(
            f"Data sequence length {data.shape[1]} does not match config {seq_len}. "
            f"Using data sequence length."
        )
        seq_len = data.shape[1]

    if data.shape[2] != n_features:
        logger.warning(
            f"Data features {data.shape[2]} does not match config {n_features}. "
            f"Using data features."
        )
        n_features = data.shape[2]

    # Create dataset
    dataset = PreNYSequenceDataset(data)

    # Split into train, validation, test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    logger.info(
        f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> VAE:
    """Train the VAE model.

    Parameters
    ----------
    model : VAE
        The VAE model to train.
    train_loader : DataLoader
        DataLoader for training data.
    val_loader : DataLoader
        DataLoader for validation data.
    config : dict
        Configuration dictionary containing training parameters.
    device : torch.device
        Device to train on (cpu or cuda).

    Returns
    -------
    VAE
        Trained VAE model.
    """
    # Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    loss_fn = VAELoss(
        reconstruction_weight=config.get("reconstruction_weight", 1.0),
        kl_weight=config.get("kl_weight", 1.0),
    )

    # Move model to device
    model.to(device)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(config["epochs"]):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)

            # Forward pass
            x_recon, mu, log_var = model(batch)

            # Compute loss
            loss = loss_fn(batch, x_recon, mu, log_var)
            train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_recon, mu, log_var = model(batch)
                loss = loss_fn(batch, x_recon, mu, log_var)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch + 1}/{config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config["output_dir"] / "vae_best.pth")
            logger.info(f"New best model saved with val_loss={val_loss:.4f}")

    return model


def evaluate_vae(model: VAE, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate the VAE model on test data.

    Parameters
    ----------
    model : VAE
        The trained VAE model.
    test_loader : DataLoader
        DataLoader for test data.
    device : torch.device
        Device to evaluate on.

    Returns
    -------
    float
        Test loss value.
    """
    model.eval()
    loss_fn = VAELoss()
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            x_recon, mu, log_var = model(batch)
            loss = loss_fn(batch, x_recon, mu, log_var)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")

    return test_loss


def main() -> None:
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train VAE for market narrative embedding")
    parser.add_argument(
        "--config", type=str, default="config/vae.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create output directory
    output_dir = Path(config.get("output_dir", "output/vae"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    # Prepare data
    logger.info("Loading and preparing data...")
    train_dataset, val_dataset, test_dataset = prepare_data(config)

    # Create data loaders
    batch_size = config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    logger.info("Initializing VAE model...")
    model = VAE(
        seq_len=config["seq_len"],
        n_features=config["n_features"],
        latent_dim=config["latent_dim"],
        encoder_channels=tuple(config.get("encoder_channels", [64, 32, 16])),
        decoder_channels=tuple(config.get("decoder_channels", [16, 32, 64])),
        kernel_size=config.get("kernel_size", 3),
        stride=config.get("stride", 2),
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    logger.info("Starting training...")
    model = train_vae(model, train_loader, val_loader, config, device)

    # Evaluate
    logger.info("Evaluating on test set...")
    evaluate_vae(model, test_loader, device)

    # Save final model
    torch.save(model.state_dict(), output_dir / "vae_final.pth")
    logger.info(f"Training complete. Final model saved to {output_dir / 'vae_final.pth'}")

    # Save configuration
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    main()
