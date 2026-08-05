from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    embed_dim: int = 128
    num_heads: int = 8
    num_object_encoder_layers: int = 3
    num_cls_layers: int = 1
    dropout: float = 0.1
    pair_input_dim: int = 4
    num_classes: int = 3
    num_object_ids: int = 7
    use_swiglu: bool = True
    use_object_id_embedding: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = x.chunk(2, dim=-1)
        return value * torch.nn.functional.silu(gate)


class FFNBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, use_swiglu: bool = True) -> None:
        super().__init__()
        hidden = dim * 4 * (2 if use_swiglu else 1)
        final_hidden = dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            SwiGLU() if use_swiglu else nn.SiLU(),
            RMSNorm(final_hidden),
            nn.Linear(final_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InputProcess(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(RMSNorm(input_dim), nn.Linear(input_dim, embed_dim), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pairwise_hmumu_features(fourvec: torch.Tensor) -> torch.Tensor:
    pt = fourvec[..., 0].clamp_min(0.0)
    eta = fourvec[..., 1].clamp(min=-10.0, max=10.0)
    phi = fourvec[..., 2]
    energy = fourvec[..., 3].clamp_min(0.0)
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    deta = eta[:, :, None] - eta[:, None, :]
    dphi = torch.atan2(torch.sin(phi[:, :, None] - phi[:, None, :]), torch.cos(phi[:, :, None] - phi[:, None, :]))
    dr = torch.sqrt(torch.clamp(deta.pow(2) + dphi.pow(2), min=0.0))
    eij = energy[:, :, None] + energy[:, None, :]
    pxij = px[:, :, None] + px[:, None, :]
    pyij = py[:, :, None] + py[:, None, :]
    pzij = pz[:, :, None] + pz[:, None, :]
    mass2 = torch.clamp(eij.pow(2) - pxij.pow(2) - pyij.pow(2) - pzij.pow(2), min=0.0)
    return torch.stack([dr, torch.abs(deta), torch.cos(dphi), torch.log1p(mass2)], dim=1)


def masked_attention_bias_value(dtype: torch.dtype) -> float:
    """A finite additive-attention sentinel that is safe in reduced precision.

    torch.finfo(float16).min is -65504, i.e. already at the edge of the representable
    range. MultiheadAttention *adds* the bias to the scaled scores, so any further
    negative contribution overflows to -inf and the softmax backward produces NaN. That
    is what disabled AMP in the previous iteration of this study.

    -1e4 is large enough that exp(-1e4) underflows to exactly 0 after the softmax shift
    (so masked keys still receive zero attention) while leaving ~6x headroom in fp16.
    """
    if dtype in (torch.float16, torch.bfloat16):
        return -1.0e4
    return -1.0e9


class PairEmbed(nn.Module):
    def __init__(self, input_dim: int, num_heads: int) -> None:
        super().__init__()
        hidden = 3 * num_heads
        self.num_heads = num_heads
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.SiLU(),
            nn.Conv1d(input_dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, num_heads, kernel_size=1),
        )

    def forward(self, fourvec: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        pair = pairwise_hmumu_features(fourvec)
        batch, _, seq, _ = pair.shape
        out = self.net(pair.reshape(batch, pair.shape[1], seq * seq))
        out = out.reshape(batch, self.num_heads, seq, seq)
        if padding_mask is not None:
            invalid = padding_mask[:, None, None, :].expand(-1, self.num_heads, seq, -1)
            out = out.masked_fill(invalid, masked_attention_bias_value(out.dtype))
        return out.reshape(batch * self.num_heads, seq, seq)


class ObjectEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, use_swiglu: bool) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = FFNBlock(dim, dropout=dropout, use_swiglu=use_swiglu)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        q = self.norm1(x)
        attn_out, _ = self.attn(q, q, q, key_padding_mask=None, attn_mask=attn_bias, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class CLSAttentionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, use_swiglu: bool) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = FFNBlock(dim, dropout=dropout, use_swiglu=use_swiglu)

    def forward(self, cls: torch.Tensor, objects: torch.Tensor, global_token: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        context = torch.cat([cls, objects, global_token], dim=1)
        batch = cls.shape[0]
        cls_global_pad = torch.zeros(batch, 2, dtype=torch.bool, device=padding_mask.device)
        key_padding = torch.cat([cls_global_pad[:, :1], padding_mask, cls_global_pad[:, 1:]], dim=1)
        norm_context = self.norm1(context)
        q = norm_context[:, :1, :]
        attn_out, _ = self.attn(q, norm_context, norm_context, key_padding_mask=key_padding, need_weights=False)
        cls = cls + self.drop1(attn_out)
        cls = cls + self.ffn(self.norm2(cls))
        return cls


class MinimalHMuMuTransformer(nn.Module):
    def __init__(self, config: ModelConfig | dict | None = None) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()
        elif isinstance(config, dict):
            config = ModelConfig(**{k: v for k, v in config.items() if k in ModelConfig.__annotations__})
        self.config = config
        self.object_input = InputProcess(5, config.embed_dim)
        self.global_input = InputProcess(7, config.embed_dim)
        self.object_id_embedding = nn.Embedding(config.num_object_ids, config.embed_dim, padding_idx=0)
        self.use_object_id_embedding = config.use_object_id_embedding
        self.pair_embed = PairEmbed(config.pair_input_dim, config.num_heads)
        self.object_layers = nn.ModuleList([
            ObjectEncoderLayer(config.embed_dim, config.num_heads, config.dropout, config.use_swiglu)
            for _ in range(config.num_object_encoder_layers)
        ])
        self.cls_layers = nn.ModuleList([
            CLSAttentionLayer(config.embed_dim, config.num_heads, config.dropout, config.use_swiglu)
            for _ in range(config.num_cls_layers)
        ])
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.embed_dim))
        self.cls_norm = RMSNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, object_features: torch.Tensor, global_features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        continuous = object_features[:, :, :5]
        object_ids = object_features[:, :, 5].long().clamp(min=0, max=self.config.num_object_ids - 1)
        fourvec = object_features[:, :, 6:10]
        if padding_mask is None:
            padding_mask = fourvec[:, :, 0] <= 0.0
        x = self.object_input(continuous)
        if self.use_object_id_embedding:
            x = x + self.object_id_embedding(object_ids)
        attn_bias = self.pair_embed(fourvec, padding_mask)
        for layer in self.object_layers:
            x = layer(x, padding_mask=padding_mask, attn_bias=attn_bias)
        global_token = self.global_input(global_features).unsqueeze(1)
        global_id = torch.full((global_features.shape[0], 1), 6, dtype=torch.long, device=global_features.device)
        if self.use_object_id_embedding:
            global_token = global_token + self.object_id_embedding(global_id)
        cls = self.cls_token.expand(global_features.shape[0], -1, -1)
        for layer in self.cls_layers:
            cls = layer(cls, x, global_token, padding_mask)
        logits = self.head(self.cls_norm(cls.squeeze(1)))
        return logits

    @torch.inference_mode()
    def predict_proba(self, object_features: torch.Tensor, global_features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return torch.softmax(self.forward(object_features, global_features, padding_mask), dim=1)


def build_model(config: dict | None = None) -> MinimalHMuMuTransformer:
    return MinimalHMuMuTransformer(config or {})
