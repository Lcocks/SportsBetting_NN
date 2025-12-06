import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

@dataclass
class StatSeqConfig:

    n_past_games: int = 5
    batch_size: int = 64
    hidden_size: int = 128
    n_epochs: int = 20
    lr: float = 1e-3

@dataclass
class TrainConfig:

    n_epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "auto"  
    verbose: bool = True

class LSTMCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

        with torch.no_grad():
            nn.init.xavier_uniform_(self.W.weight)
            nn.init.zeros_(self.W.bias)
            self.W.bias[:hidden_size] = 1.0

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        z = self.W(torch.cat([x_t, h_prev], dim=1))  # (B, 4H)
        H = self.hidden_size
        f, i, o, g = z.chunk(4, dim=1)

        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class LSTMSequence(nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:

        B, T, E = x.shape
        H = self.cell.hidden_size

        # initialize hidden & cell
        h = x.new_zeros(B, H)
        c = x.new_zeros(B, H)

        # no masking yet; assumes full sequences
        for t in range(T):
            x_t = x[:, t, :]      # (B, E)
            h, c = self.cell(x_t, h, c)

        return h  # (B, H)


class DualHeadStatModel(nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.sequence = LSTMSequence(input_size, hidden_size)

        # Regression head
        self.fc_reg = nn.Linear(hidden_size, 1)
        # Classification head
        self.fc_bin = nn.Linear(hidden_size, 1)

        with torch.no_grad():
            nn.init.zeros_(self.fc_reg.bias)
            nn.init.zeros_(self.fc_bin.bias)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        h_last = self.sequence(x, lengths)      

        y_reg = self.fc_reg(h_last).squeeze(-1)      
        logits_bin = self.fc_bin(h_last).squeeze(-1) 

        return y_reg, logits_bin


def build_lstm_model(
    input_size: int,
    hidden_size: int = 128,
) -> nn.Module:

    return DualHeadStatModel(input_size, hidden_size)


def _resolve_device(device_str: str) -> torch.device:

    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_str)


def train_dual_head_classifier(
    X: Any,
    y: Any,
    lengths: Any,
    hidden_size: int = 128,
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

    model = build_lstm_model(input_size=E, hidden_size=hidden_size)
    model = model.to(device)

    dataset = TensorDataset(X_t, y_t, lengths_t)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        total_loss = 0.0

        for X_b, y_b, len_b in dataloader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            len_b = len_b.to(device)

            optimizer.zero_grad()

            y_reg, logits = model(X_b, len_b) 

            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y_b.size(0)

        avg_loss = total_loss / N
        history.append({"epoch": epoch, "train_bce": avg_loss})

        if cfg.verbose:
            print(f"Epoch {epoch:02d} | Train BCE loss: {avg_loss:.4f}")

    return {
        "model": model,
        "history": history,
    }