import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

@dataclass
class TrainConfig:

    n_epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "auto"   # "auto", "cpu", or "cuda"
    verbose: bool = True


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        T = x.size(0)
        return x + self.pe[:T]

class SimpleTFTBackbone(nn.Module):

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=False,  # we will use (T, B, D)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.pos_encoder = PositionalEncoding(d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:

        B, T, E = x.shape

        x_proj = self.input_proj(x) 

        x_proj = x_proj.transpose(0, 1)  

        x_pe = self.pos_encoder(x_proj)  

        enc_out = self.encoder(x_pe)   

        h_last = enc_out[-1]          

        return h_last

class DualHeadTFTModel(nn.Module):

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = SimpleTFTBackbone(
            input_size=input_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Regression head
        self.fc_reg = nn.Linear(d_model, 1)
        # Classification head
        self.fc_bin = nn.Linear(d_model, 1)

        with torch.no_grad():
            nn.init.zeros_(self.fc_reg.bias)
            nn.init.zeros_(self.fc_bin.bias)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        h_last = self.backbone(x, lengths)      # (B, d_model)

        y_reg = self.fc_reg(h_last).squeeze(-1)      # (B,)
        logits_bin = self.fc_bin(h_last).squeeze(-1) # (B,)

        return y_reg, logits_bin


def build_tft_model(
    input_size: int,
    d_model: int = 128,
    n_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> nn.Module:

    return DualHeadTFTModel(
        input_size=input_size,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout,
    )


def _resolve_device(device_str: str) -> torch.device:
    """
    Helper to resolve the device based on a string.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_str)


def train_tft_classifier(
    X: Any,
    y: Any,
    lengths: Any,
    d_model: int = 128,
    n_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    cfg: TrainConfig = TrainConfig(),
) -> Dict[str, Any]:

    device = _resolve_device(cfg.device)

    # Convert inputs to tensors if needed
    if not isinstance(X, torch.Tensor):
        X_t = torch.tensor(X, dtype=torch.float32)
    else:
        X_t = X.float()

    if not isinstance(y, torch.Tensor):
        y_t = torch.tensor(y, dtype=torch.float32)
    else:
        y_t = y.float()

    if not isinstance(lengths, torch.Tensor):
        lengths_t = torch.tensor(lengths, dtype=torch.long)
    else:
        lengths_t = lengths.long()

    N, T, E = X_t.shape

    # Build model
    model = build_tft_model(
        input_size=E,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
    model = model.to(device)

    # Dataset & DataLoader
    dataset = TensorDataset(X_t, y_t, lengths_t)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    history: List[Dict[str, float]] = []

    # Training loop
    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        total_loss = 0.0

        for X_b, y_b, len_b in dataloader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            len_b = len_b.to(device)

            optimizer.zero_grad()

            # Dual-head forward: ignore regression head in loss
            y_reg, logits = model(X_b, len_b)  # y_reg unused here

            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y_b.size(0)

        avg_loss = total_loss / N
        history.append({"epoch": epoch, "train_bce": avg_loss})

        if cfg.verbose:
            print(f"[TFT] Epoch {epoch:02d} | Train BCE loss: {avg_loss:.4f}")

    return {
        "model": model,
        "history": history,
    }