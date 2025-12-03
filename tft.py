from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

@dataclass
class TFTConfig:
    hidden_size: int = 64
    num_heads: int = 2
    dropout: float = 0.1

    n_past: int = 12               
    n_future: int = 4           
    static_cardinalities: List[int] = None 

    n_epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3

    device: str = "cpu"


class GatedResidualNetwork(nn.Module):
    def __init__(self, inp: int, hidden: int, out: int | None = None, dropout: float = 0.1):
        super().__init__()
        out = out or inp

        self.fc1 = nn.Linear(inp, hidden)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden, out)
        self.dropout = nn.Dropout(dropout)

        # gating
        self.fc_gate = nn.Linear(out, out)
        self.sigmoid = nn.Sigmoid()

        # skip
        if inp != out:
            self.skip = nn.Linear(inp, out)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)

        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        gate = self.sigmoid(self.fc_gate(x))
        return residual + gate * x


class VariableSelectionNetwork(nn.Module):

    def __init__(self, n_inputs: int, d_inp: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.n_inputs = n_inputs

        self.weight_grn = GatedResidualNetwork(
            n_inputs * d_inp,
            hidden_size,
            out=n_inputs,
            dropout=dropout,
        )
        self.softmax = nn.Softmax(dim=-1)

        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(d_inp, hidden_size, out=hidden_size, dropout=dropout)
            for _ in range(n_inputs)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, n_vars, d = x.shape

        flat = x.reshape(B, T, n_vars * d)
        w = self.weight_grn(flat)              # (B,T,n_vars)
        w = self.softmax(w).unsqueeze(-1)      # (B,T,n_vars,1)

        var_outs = []
        for i, grn in enumerate(self.var_grns):
            v = x[:, :, i, :]                  # (B,T,d_inp)
            var_outs.append(grn(v))            # (B,T,hidden)
        var_outs = torch.stack(var_outs, dim=2)  # (B,T,n_vars,hidden)

        z = torch.sum(w * var_outs, dim=2)     # (B,T,hidden)
        return z



class TemporalFusionTransformer(nn.Module):


    def __init__(
        self,
        hidden_size: int,
        n_static_cat: int,
        n_future: int,
        n_past: int,
        static_cardinalities: List[int],
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert len(static_cardinalities) == n_static_cat, (
            "len(static_cardinalities) must equal n_static_cat"
        )

        self.hidden_size = hidden_size
        self.n_static_cat = n_static_cat
        self.n_future = n_future
        self.n_past = n_past

        # Embeddings for static categorical variables
        self.static_emb_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=hidden_size)
            for card in static_cardinalities
        ])

        # Static variable selection
        self.static_vsn = VariableSelectionNetwork(
            n_inputs=n_static_cat,
            d_inp=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        # Past observed VSN
        self.past_vsn = VariableSelectionNetwork(
            n_inputs=n_past,
            d_inp=1,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        # Known future VSN
        self.future_vsn = VariableSelectionNetwork(
            n_inputs=n_future,
            d_inp=1,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        # LSTM encoder/decoder
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Final head
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        past_obs: torch.Tensor,   # (B,T,n_past)
        fut_known: torch.Tensor,  # (B,1,n_future)
        static_cat: torch.Tensor  # (B,n_static_cat)
    ) -> torch.Tensor:

        B, T, _ = past_obs.shape

        # ---- Static ----
        static_embs = []
        for i, emb in enumerate(self.static_emb_layers):
            static_embs.append(emb(static_cat[:, i]))  # (B,hidden)
        static_embs = torch.stack(static_embs, dim=1)   # (B,S_static,hidden)

        static_repr = self.static_vsn(static_embs.unsqueeze(1))  # (B,1,hidden)
        static_repr = static_repr.squeeze(1)                     # (B,hidden)
        # static_repr currently unused, but kept for extensibility

        # ---- Past observed ----
        past_obs_exp = past_obs.unsqueeze(-1)  # (B,T,n_past,1)
        z_past = self.past_vsn(past_obs_exp)   # (B,T,hidden)

        # ---- Known future ----
        fut_known_exp = fut_known.unsqueeze(-1)  # (B,1,n_future,1)
        z_fut = self.future_vsn(fut_known_exp)   # (B,1,hidden)

        # ---- LSTM encoder/decoder ----
        enc_out, (h, c) = self.encoder(z_past)   # enc_out: (B,T,H)
        dec_out, _ = self.decoder(z_fut, (h, c)) # dec_out: (B,1,H)

        # ---- Attention over encoder output ----
        attn_out, _ = self.attn(dec_out, enc_out, enc_out)  # (B,1,H)

        logits = self.fc(attn_out).squeeze(1).squeeze(-1)   # (B,)
        return logits


# ============================================================
# 4. FACTORY + EVAL HELPER
# ============================================================

def build_tft_model(
    cfg: TFTConfig,
    n_static_cat: int,
    n_future: int,
    n_past: int,
    static_cardinalities: List[int],
    device: torch.device | str = "cpu",
) -> TemporalFusionTransformer:

    model = TemporalFusionTransformer(
        hidden_size=cfg.hidden_size,
        n_static_cat=n_static_cat,
        n_future=n_future,
        n_past=n_past,
        static_cardinalities=static_cardinalities,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    ).to(device)
    return model


@torch.no_grad()
def evaluate_tft_binary(
    model: nn.Module,
    dataloader,
    device: torch.device | str = "cpu",
):

    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for past_obs, fut_known, static_cat, y in dataloader:
        past_obs   = past_obs.to(device)
        fut_known  = fut_known.to(device)
        static_cat = static_cat.to(device)
        y          = y.to(device)

        logits = model(past_obs, fut_known, static_cat)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total_loss += loss.item() * y.size(0)
        total_correct += (preds == y).sum().item()
        total_examples += y.numel()

    avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
    acc = total_correct / total_examples if total_examples > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": acc,
    }