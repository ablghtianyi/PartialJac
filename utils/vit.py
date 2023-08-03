import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class FeedForward(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.dim),
            nn.Dropout(config.dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        inner_dim = config.dim_head * config.heads
        project_out = not (config.heads == 1 and config.dim_head == config.dim)

        self.heads = config.heads
        self.scale = config.dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(config.dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, config.dim),
            nn.Dropout(config.dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PreLNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mu = config.mu
        self.layers = nn.ModuleList([])
        self.ln1 = nn.LayerNorm(config.dim)
        self.ln2 = nn.LayerNorm(config.dim)
        self.attn = Attention(config)
        self.mlp = FeedForward(config)
        
    def forward(self, x):
        x = self.attn(self.ln1(x)) + self.mu * x
        x = self.mlp(self.ln2(x)) + self.mu * x
        return x


class PostLNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mu = config.mu
        self.layers = nn.ModuleList([])
        self.ln1 = nn.LayerNorm(config.dim)
        self.ln2 = nn.LayerNorm(config.dim)
        self.attn = Attention(config)
        self.mlp = FeedForward(config)
        
    def forward(self, x):
        x = self.ln1(self.attn(x) + self.mu * x)
        x = self.ln2(self.mlp(x) + self.mu * x)
        return x


class NoLNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mu = config.mu
        self.layers = nn.ModuleList([])
        self.attn = Attention(config)
        self.mlp = FeedForward(config)
        
    def forward(self, x):
        x = self.attn(x) + self.mu * x
        x = self.mlp(x) + self.mu * x
        return x
    
    
class PreLNAttnBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mu = config.mu
        self.layers = nn.ModuleList([])
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = Attention(config)
        
    def forward(self, x):
        x = self.attn(self.ln1(x)) + self.mu * x
        return x
    
    
class PostLNAttnBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mu = config.mu
        self.layers = nn.ModuleList([])
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = Attention(config)
        
    def forward(self, x):
        x = self.ln1(self.attn(x) + self.mu * x)
        return x


class NoLNAttnBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mu = config.mu
        self.layers = nn.ModuleList([])
        self.attn = Attention(config)
        
    def forward(self, x):
        x = self.attn(x) + self.mu * x
        return x
    
    
class ViT(nn.Module):
    def __init__(self, Block_Module: nn.Module, config):
        super().__init__()
        image_height, image_width = pair(config.image_size)
        patch_height, patch_width = pair(config.patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = config.channels * patch_height * patch_width
        assert config.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, config.dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, config.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim))
        self.dropout = nn.Dropout(config.emb_dropout)

        self.transformer = nn.ModuleList([Block_Module(config) for _ in range(config.depth)])

        self.pool = config.pool
        # self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.num_classes)
        )
        self._reset_parameters(config)
    
    def _reset_parameters(self, config):
        for id, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                m.weight.data *= config.varw**0.5
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean = 0.0, std = config.varb**0.5)
        
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        for m in self.transformer:
            x = m(x)
            
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)
        return self.mlp_head(x)