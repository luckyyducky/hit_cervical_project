
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import MLP
from torchvision.models.vision_transformer import MLPBlock
from collections import OrderedDict
from functools import partial
from typing import Callable
from core.model.modules.attention import MultiheadAttention
from timm.models.layers import DropPath

class EncoderBlockHiT(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, mlp_ratio: int, dropout: float, attention_dropout: float, mlp_dropout: float, norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), drop_path: float = 0.0, remove_mlp: bool = False):
        super().__init__()
        self.num_heads = num_heads
        mlp_dim = int(mlp_ratio * hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout) if drop_path > 0. else nn.Identity()
        if not remove_mlp:
            self.ln_2 = norm_layer(hidden_dim)
            self.mlp = MLPBlock(hidden_dim, mlp_dim, mlp_dropout)
        else:
            self.mlp = None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input: torch.Tensor, class_token: torch.Tensor, get_additional_outputs: bool = False):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        kv = self.ln_1(input)
        q = self.ln_1(class_token)
        q, attn_map, vectors = self.self_attention(q, kv, kv, need_weights=get_additional_outputs)
        class_token = class_token + self.drop_path(self.dropout(q))
        if self.mlp is not None:
            input = self.drop_path(self.dropout(self.mlp(self.ln_2(input)))) + input
        other_outputs = {'attn_map': attn_map, 'vectors': vectors}
        return input, class_token, other_outputs

def get_block_version(version: str = 'hit'):
    if version in ['hit', '2.2', '2.3']:
        return EncoderBlockHiT
    else:
        raise ValueError(f"Unknown block version: {version}")

class PoolingTokens(nn.Module):
    def __init__(self, factor, in_dim, out_dim):
        super().__init__()
        self.factor = factor
        self.avg_pool = nn.AvgPool2d(kernel_size=factor)
        self.linear = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, input, class_token, get_additional_outputs=False):
        if self.factor == 1:
            return self.linear(input), self.linear(class_token), None
        B, S, C = input.shape
        sS = int(math.sqrt(S))
        input = input.permute((0, 2, 1)).view(B, C, sS, sS)
        input = self.avg_pool(input)
        input = input.view(B, C, -1).permute(0, 2, 1)
        return self.linear(input), self.linear(class_token), None

class SpatialEncoder(nn.Module):
    def __init__(self, seq_length, num_heads, mlp_ratio, dropout, attention_dropout, mlp_dropout, drop_path, norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), block_version: str = '2', ln_at_the_end: bool = True, blocks: list = [4, 4, 12, 4], block_dims: list = [64, 128, 320, 512]):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, block_dims[0]).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        EncoderBlock = get_block_version(block_version)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        last_block_hidden_dim = block_dims[0]
        for idx, (num_layers, hidden_dim) in enumerate(zip(blocks, block_dims)):
            layer = OrderedDict()
            layer['pool'] = PoolingTokens(2 if idx != 0 else 1, last_block_hidden_dim, hidden_dim)
            last_block_hidden_dim = hidden_dim
            for jdx in range(num_layers):
                layer[f'encoder_layer_{jdx}'] = EncoderBlock(num_heads=num_heads, hidden_dim=hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout, attention_dropout=attention_dropout, mlp_dropout=mlp_dropout, norm_layer=norm_layer, drop_path=drop_path, remove_mlp=block_version == '2.2' and (jdx == (num_layers - 1)) and (idx == (len(blocks) - 1)))
            layers[f"block_{idx}"] = nn.Sequential(layer)
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim) if ln_at_the_end else None

    def forward(self, x, class_token, get_additional_outputs=False):
        if self.pos_embedding is not None:
            x = x + self.pos_embedding[:, 1:]
            class_token = class_token + self.pos_embedding[:, :1]
        for idx, module in enumerate(self.layers):
            for jdx, m in enumerate(module):
                x, class_token, _ = m(input=x, class_token=class_token, get_additional_outputs=get_additional_outputs)
        if self.ln is not None:
            class_token = self.ln(class_token)
        return x, class_token

class SpatialInterpretableTransformer(nn.Module):
    def __init__(self, image_size: int, patch_size: int, num_heads: int, mlp_ratio: int, dropout: float = 0.0, attention_dropout: float = 0.0, mlp_dropout: float = 0.0, drop_path: float = 0.0, num_classes: int = 1000, use_pos_tokens: bool = True, norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), block_version: str = '2.3', ln_at_the_end: bool = True, blocks=[4, 4, 12, 4], block_dims=[64, 128, 320, 512]):
        super().__init__()
        seq_length = (image_size // patch_size) ** 2 + 1
        self.class_token = nn.Parameter(torch.zeros(1, 1, block_dims[0]))
        self.encoder = SpatialEncoder(seq_length=seq_length if use_pos_tokens else -1, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, attention_dropout=attention_dropout, mlp_dropout=mlp_dropout, drop_path=drop_path, norm_layer=norm_layer, block_version=block_version, ln_at_the_end=ln_at_the_end, blocks=blocks, block_dims=block_dims)
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers['head'] = nn.Linear(block_dims[-1], num_classes, bias=True)
        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=block_dims[0], kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size
        self.image_size = image_size
        self.hidden_dim = block_dims[0]
        self.num_classes = num_classes
        self.heads = nn.Sequential(heads_layers)
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)
        nn.init.zeros_(self.heads.head.weight)
        nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p
        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x, return_features=False, get_additional_outputs=False):
        x = self._process_input(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x, class_token = self.encoder(x=x, class_token=class_token, get_additional_outputs=get_additional_outputs)
        class_token = class_token.squeeze(dim=1)
        if return_features:
            return self.heads(class_token), x
        return self.heads(class_token)
