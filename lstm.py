import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List
import numpy as np


# ============================================================
# 1. CONFIG
# ============================================================

@dataclass
class StatSeqConfig:
    n_past_games: int = 5
    batch_size: int = 64
    hidden_size: int = 128
    n_epochs: int = 20
    lr: float = 1e-3


# ============================================================
# 2. LSTM BUILDING BLOCKS
# ============================================================

class LSTMCell(nn.Module):
    """
    A custom implementation of an LSTM cell (not PyTorch's built-in).
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

        with torch.no_grad():
            nn.init.xavier_uniform_(self.W.weight)
            nn.init.zeros_(self.W.bias)
            # forget gate bias trick
            self.W.bias[:hidden_size] = 1.0

    def forward(self, x_t, h_prev, c_prev):
        z = self.W(torch.cat([x_t, h_prev], dim=1))
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
    """
    Runs the custom LSTMCell over a sequence.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x, lengths):
        """
        x:       (B, T, E)
        lengths: (B,)
        """
        B, T, E = x.shape
        H = self.cell.hidden_size

        h = x.new_zeros(B, H)
        c = x.new_zeros(B, H)

        for t in range(T):
            x_t = x[:, t, :]
            h, c = self.cell(x_t, h, c)  # no masking needed unless using padded batches

        return h  # final hidden state


# ============================================================
# 3. LSTM MODELS (REGRESSION + BINARY)
# ============================================================

class StatFromScratch(nn.Module):
    """
    Regression version (predicts a continuous stat: yards, receptions, etc.)
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.sequence = LSTMSequence(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

        nn.init.zeros_(self.fc.bias)

    def forward(self, x, lengths):
        h_last = self.sequence(x, lengths)
        out = self.fc(h_last).squeeze(-1)
        return out


class StatFromScratchBinary(nn.Module):
    """
    Binary classification version (over/under threshold).
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.sequence = LSTMSequence(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

        nn.init.zeros_(self.fc.bias)

    def forward(self, x, lengths):
        h_last = self.sequence(x, lengths)
        logits = self.fc(h_last).squeeze(-1)
        return logits


def build_lstm_model(input_size: int, hidden_size: int = 128, binary: bool = True):
    """
    Creates an untrained LSTM model.
    """
    if binary:
        model = StatFromScratchBinary(input_size, hidden_size)
    else:
        model = StatFromScratch(input_size, hidden_size)

    return model