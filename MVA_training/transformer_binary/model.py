"""
Binary ggH-vs-VBF ParticleTransformer (architecture only), copied from Xudong Hu's b-hive-h-mu-mu
(gitlab.cern.ch/xudongh/b-hive-h-mu-mu, utils/models/particletransformer2.py at d57c342).
Same network and state_dict (strict load); only forward() takes named tensors and unused paths are dropped.
Output: one logit, sigmoid = ggH probability, so VBF = score < cutoff.
"""

import copy
import math
from functools import partial

import torch
import torch.nn as nn

# Object-token layout, per event: mu1, mu2, jet1, jet2, dimuon (padding id 0, global token id 6)
OBJECTS = ("mu1", "mu2", "jet1", "jet2", "dimuon")
OBJECT_IDS = {"mu1": 1, "mu2": 2, "jet1": 3, "jet2": 4, "dimuon": 5}
GLOBAL_TOKEN_ID = 6
# per-object features: 5 continuous, the object id, then (pt, eta, phi, E) for the pair features
OBJECT_FEATURES = (
    "ln_pt", "ln_e", "eta", "sin_phi", "cos_phi", "id", "pt_4v", "eta_4v", "phi_4v", "energy_4v",
)
GLOBAL_FEATURES = (
    "ln_htsoft2", "ln_htsoft5", "MET_ln_pt", "MET_ln_e", "MET_sin_phi", "MET_cos_phi", "n_jets",
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * self.act(x2)


class FFNBlock(nn.Module):
    def __init__(self, dim, swiglu=True, dropout=0.1):
        super().__init__()
        self.dense_1 = nn.Linear(dim, dim * 8 if swiglu else dim * 4)
        self.activation = SwiGLU() if swiglu else nn.SiLU()
        self.dense_2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_norm = RMSNorm(dim * 4)

    def forward(self, x):
        x = self.attn_norm(self.activation(self.dense_1(x)))
        return self.dropout(self.dense_2(x))


def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def pairwise_hmumu_fts(xi, xj, eps=1e-8):
    """Pair features from (pt, eta, phi, E): DeltaR, |Delta eta|, cos(Delta phi), ln(1+m^2)."""
    pti, etai, phii, ei = xi[:, :4].split((1, 1, 1, 1), dim=1)
    ptj, etaj, phij, ej = xj[:, :4].split((1, 1, 1, 1), dim=1)
    pxi, pyi, pzi = pti * torch.cos(phii), pti * torch.sin(phii), pti * torch.sinh(etai)
    pxj, pyj, pzj = ptj * torch.cos(phij), ptj * torch.sin(phij), ptj * torch.sinh(etaj)

    mask = torch.ne(pti, 0.0).to(xi.dtype) * torch.ne(ptj, 0.0).to(xi.dtype)
    deta = etai - etaj
    dphi = delta_phi(phii, phij)
    delta_r = (deta.square() + dphi.square()).clamp(min=0).sqrt()
    m2 = (ei + ej).square() - (pxi + pxj).square() - (pyi + pyj).square() - (pzi + pzj).square()
    lnm2 = torch.log(m2.clamp(min=eps) + 1)
    return torch.cat([delta_r, deta.abs(), torch.cos(dphi), lnm2], dim=1) * mask


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # same draw as b-hive (timm's implementation), so a given seed initialises identically
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        low, up = norm_cdf((a - mean) / std), norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class PairEmbed(nn.Module):
    """Pairwise attention bias. `masked_lnm2_pairs` zeroes ln(1+m^2) for the listed token
    pairs only (the mu1-mu2 pair, so the dimuon mass is not a direct input)."""

    def __init__(self, input_dim, dims, eps=1e-8, masked_lnm2_pairs=()):
        super().__init__()
        self.pairwise_fts = partial(pairwise_hmumu_fts, eps=eps)
        self.masked_lnm2_pairs = tuple(tuple(sorted(map(int, pair))) for pair in masked_lnm2_pairs)
        module_list = []
        for dim in dims:
            module_list.extend([nn.BatchNorm1d(input_dim), nn.SiLU(), nn.Conv1d(input_dim, dim, 1)])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)
        self.out_dim = dims[-1]

    def forward(self, x):
        batch_size, _, seq_len = x.size()
        i, j = torch.tril_indices(seq_len, seq_len, offset=-1, device=x.device)
        x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
        x = self.pairwise_fts(x[:, :, i, j], x[:, :, j, i])
        if self.masked_lnm2_pairs:
            pair_mask = torch.zeros_like(i, dtype=torch.bool)
            for first, second in self.masked_lnm2_pairs:
                pair_mask |= ((i == second) & (j == first)) | ((i == first) & (j == second))
            lnm2_keep = (~pair_mask).to(dtype=x.dtype).reshape(1, 1, -1)
            x = torch.cat((x[:, :3], x[:, 3:4] * lnm2_keep), dim=1)
        elements = self.embed(x)
        y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=x.device)
        y[:, :, i, j] = elements
        y[:, :, j, i] = elements
        return y


class InputConv(nn.Module):
    def __init__(self, in_chn, out_chn, dropout_rate=0.1):
        super().__init__()
        self.lin = nn.Linear(in_chn, out_chn)
        self.bn1 = RMSNorm(in_chn)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sc, skip=True):
        x2 = self.dropout(self.act(self.lin(self.bn1(x))))
        return sc + x2 if skip else x2


class InputProcess_JC(nn.Module):
    def __init__(self, cpf_dim, embed_dim):
        super().__init__()
        self.cpf_conv1 = InputConv(cpf_dim, embed_dim)
        self.cpf_conv3 = InputConv(embed_dim, embed_dim)

    def forward(self, cpf):
        cpf = self.cpf_conv1(cpf, cpf, skip=False)
        return self.cpf_conv3(cpf, cpf, skip=True)


class HF_TransformerEncoderLayer(nn.Module):
    """Pre-norm self-attention over the object tokens, with the pair bias as attn_mask."""

    def __init__(self, d_model, nhead, dropout=0.1, swiglu=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FFNBlock(d_model, swiglu=swiglu, dropout=dropout)
        self.activation = nn.SiLU()

    def forward(self, src, mask):
        normed = self.attn_norm(src)
        src = src + self.self_attn(normed, normed, normed, attn_mask=mask, need_weights=False)[0]
        return src + self.ffn(self.ffn_norm(src))


class HF_TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # deep copies of one layer: every layer starts from the same initial weights, as in b-hive
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask):
        for layer in self.layers:
            src = layer(src, mask)
        return src


class CLS_TransformerEncoderLayer(nn.Module):
    """Class attention: the CLS token attends to itself, the objects and the global token."""

    def __init__(self, d_model, nhead, dropout=0.1, swiglu=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.cls_norm = RMSNorm(d_model)
        self.ffn = FFNBlock(d_model, swiglu=swiglu, dropout=dropout)
        self.activation = nn.SiLU()

    def forward(self, cls_token, x, padding_mask):
        src = self.attn_norm(torch.cat((cls_token, x), dim=1))
        padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
        attn_out = self.self_attn(
            self.cls_norm(cls_token), src, src, key_padding_mask=padding_mask, need_weights=False
        )[0]
        src = cls_token + attn_out
        return src + self.ffn(self.ffn_norm(src))


class BinaryHmumuTransformer(nn.Module):
    """ParticleTransformer_HMuMu_GgHVsVBF_w_wgt. Defaults are the Sep08 production model."""

    def __init__(
        self,
        n_years,
        num_enc=3,
        num_head=8,
        embed_dim=128,
        dropout=0.1,
        swiglu=True,
        mask_mu1_mu2_lnm2=True,
    ):
        super().__init__()
        self.hparams = dict(
            n_years=n_years, num_enc=num_enc, num_head=num_head, embed_dim=embed_dim,
            dropout=dropout, swiglu=swiglu, mask_mu1_mu2_lnm2=mask_mu1_mu2_lnm2,
        )
        self.num_enc_layers = num_enc
        self.num_head = num_head
        # construction order follows b-hive so that one seed gives one initialisation
        self.YearEmbedding = nn.Embedding(n_years, embed_dim)
        n_continuous = OBJECT_FEATURES.index("id")
        self.InputProcess = InputProcess_JC(n_continuous, embed_dim)
        self.ObjectIDEmbedding = nn.Embedding(7, embed_dim, padding_idx=0)
        self.GlobalInputProcess = InputProcess_JC(len(GLOBAL_FEATURES), embed_dim)
        self.Linear = nn.Linear(embed_dim, 1)
        self.pair_embed = PairEmbed(
            4,
            [3 * num_head] * 2 + [num_head],
            masked_lnm2_pairs=((0, 1),) if mask_mu1_mu2_lnm2 else (),
        )
        self.cls_norm = RMSNorm(embed_dim)
        self.Encoder = HF_TransformerEncoder(
            HF_TransformerEncoderLayer(embed_dim, num_head, dropout=dropout, swiglu=swiglu), num_enc
        )
        self.CLS_EncoderLayer1 = CLS_TransformerEncoderLayer(embed_dim, num_head, dropout=dropout, swiglu=swiglu)
        if num_enc > 3:
            self.CLS_EncoderLayer2 = CLS_TransformerEncoderLayer(embed_dim, num_head, dropout=dropout, swiglu=swiglu)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, global_features, year_idx, objects):
        """
        global_features: (B, 7) float, ordered as GLOBAL_FEATURES
        year_idx:        (B,) long, index into the run period's year vocabulary
        objects:         (B, 5, 10) float, tokens OBJECTS with features OBJECT_FEATURES
        returns          (B, 1) logit; sigmoid gives the ggH probability
        """
        n_tokens = objects.size(1)
        continuous = objects[:, :, : OBJECT_FEATURES.index("id")]
        object_id = objects[:, :, OBJECT_FEATURES.index("id")].long().clamp(min=0, max=6)
        four_vectors = objects[:, :, -4:]  # pt, eta, phi, E

        # an empty object slot has pt == 0; its attention logits get float16's most negative value
        padding_mask = torch.eq(four_vectors[:, :, 0], 0.0)
        pad = (padding_mask.unsqueeze(1) + padding_mask.unsqueeze(2)).to(continuous.dtype) * torch.finfo(torch.float16).min

        enc = self.InputProcess(continuous) + self.ObjectIDEmbedding(object_id)
        attn_mask = self.pair_embed(four_vectors.transpose(1, 2)) + pad.unsqueeze(dim=1)
        enc = self.Encoder(enc, attn_mask.reshape(-1, n_tokens, n_tokens))

        # MET and soft activity enter only through one global token seen by class attention
        global_token = self.GlobalInputProcess(global_features.unsqueeze(1))
        global_token = global_token + self.YearEmbedding(year_idx.long()).unsqueeze(1)
        global_id = torch.full((global_features.shape[0], 1), GLOBAL_TOKEN_ID, dtype=torch.long, device=global_features.device)
        global_token = global_token + self.ObjectIDEmbedding(global_id)

        enc_for_cls = torch.cat((enc, global_token), dim=1)
        padding_mask_for_cls = torch.cat((padding_mask, torch.zeros_like(padding_mask[:, :1])), dim=1)

        cls_tokens = self.cls_token.expand(enc.size(0), 1, -1)
        cls_tokens = self.CLS_EncoderLayer1(cls_tokens, enc_for_cls, padding_mask_for_cls)
        if self.num_enc_layers > 3:
            cls_tokens = self.CLS_EncoderLayer2(cls_tokens, enc_for_cls, padding_mask_for_cls)
        return self.Linear(self.cls_norm(torch.squeeze(cls_tokens, dim=1)))
