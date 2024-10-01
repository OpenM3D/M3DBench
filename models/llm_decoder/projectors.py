from torch import nn
from models.llm_decoder.config import CONFIG

class SceneProj(nn.Module):
    def __init__(self, scene_embd, n_embd):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(scene_embd, n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd)
        )

    @classmethod
    def build(cls, scene_embd, n_embd):
        return cls(scene_embd, n_embd)

    def forward(self, x):
        return self.proj(x)


class ClickProj(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(CONFIG['click_feat_dim'], n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd)
        )

    @classmethod
    def build(cls, n_embd):
        return cls(n_embd)

    def forward(self, x):
        return self.proj(x)


class RegionProj(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(CONFIG['region_feat_dim'], n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd)
        )

    @classmethod
    def build(cls, n_embd):
        return cls(n_embd)

    def forward(self, x):
        return self.proj(x)


class ShapeProj(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(CONFIG['shape_feat_dim'], n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd)
        )

    @classmethod
    def build(cls, n_embd):
        return cls(n_embd)

    def forward(self, x):
        return self.proj(x)


class ImageProj(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(CONFIG['image_feat_dim'], n_embd),
            nn.GELU(),
            nn.Linear(n_embd, n_embd)
        )

    @classmethod
    def build(cls, n_embd):
        return cls(n_embd)

    def forward(self, x):
        return self.proj(x)
