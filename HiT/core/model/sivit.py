
import torch.nn as nn
from core.model.IT import InterpretableVisualTransformer

class SpatialInterpretableTransformer(InterpretableVisualTransformer):
    def __init__(self, *args, **kwargs):
        # --- FIX STARTS HERE ---
        # The parent class doesn't expect these arguments, so we remove them
        # from the kwargs dictionary before calling the parent's __init__.
        kwargs.pop('block_dims', None)
        kwargs.pop('blocks', None)
        kwargs.pop('block_version', None)
        # --- FIX ENDS HERE ---
        
        super().__init__(*args, **kwargs)
        
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.num_classes)
        )

    def forward_features(self, x):
        x = self.prepare_features(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
