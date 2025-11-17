import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import torch.utils.checkpoint


# === Initialization ===

def init_weights(layer, idx):
    # Apply weight initialization based on layer type
    if isinstance(layer, RMSNorm):
        if layer.weight is not None:
            nn.init.ones_(layer.weight)
    if isinstance(layer, nn.LayerNorm):
        if layer.weight is not None:
            nn.init.ones_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    if isinstance(layer, nn.Linear):
        nn.init.trunc_normal_(layer.weight, std=0.02)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def apply(mod, fn, idx):
    # Recursively apply a function to all child modules
    for module in mod.children():
        apply(module, fn, idx)
    fn(mod, idx)
    return mod


# === RMSNorm ===

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps

        import numbers
        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = (hidden_states * self.weight).to(input_dtype)
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


# === Self-Attention with RMSNorm and Flash Attention ===

class SelfAttention(nn.Module):
    def __init__(self, hidden, num_head=16):
        super().__init__()
        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.qnorm = RMSNorm(hidden // num_head, eps=1e-5)
        self.knorm = RMSNorm(hidden // num_head, eps=1e-5)

        self.hidden = hidden
        self.head_dim = hidden // num_head
        self.num_head = num_head
        self.out_linear = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        out = self.qkv(x)
        
        q, k, v = out.split(self.hidden, dim=2)
        q = einops.rearrange(q, "b l (nh hd) -> b (l nh) hd", hd=self.head_dim)
        k = einops.rearrange(k, "b l (nh hd) -> b (l nh) hd", hd=self.head_dim)
        v = einops.rearrange(v, "b l (nh hd) -> b (l nh) hd", hd=self.head_dim)

        q = self.qnorm(q)
        k = self.knorm(k)

        q = einops.rearrange(q, "b (l nh) hd -> b nh l hd", nh=self.num_head)
        k = einops.rearrange(k, "b (l nh) hd -> b nh l hd", nh=self.num_head)
        v = einops.rearrange(v, "b (l nh) hd -> b nh l hd", nh=self.num_head)

        # https://discuss.pytorch.org/t/flash-attention/174955/14
        output = F.scaled_dot_product_attention(q, k, v)
        output = self.out_linear(einops.rearrange(output, "b nh l hd -> b l (nh hd)"))
        return output


# === Transformer Block ===

class Block(nn.Module):
    def __init__(self, hidden, linear_dim):
        super().__init__()
        self.norm_1 = nn.LayerNorm(hidden, bias=False)
        self.norm_2 = nn.LayerNorm(hidden, bias=False)
        self.in_linear = nn.Linear(hidden, linear_dim, bias=False)
        self.out_linear = nn.Linear(linear_dim, hidden, bias=False)
        self.act = nn.GELU()
        self.self_attention = SelfAttention(hidden)

    def forward(self, x):
        attention = self.self_attention(self.norm_1(x))
        x = x + attention
        linear = self.out_linear(self.act(self.in_linear(self.norm_2(x))))
        return x + linear


# === CNN-based xyz Predictor ===

class xyzPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1)
        )

        self.fc = nn.Tanh()  # Output constrained to [-1, 1]
        # self.fc = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc(x)


# === Decoder (Transformer-based) ===

class Decoder(nn.Module):
    def __init__(
        self,
        hidden,
        num_layers,
        input_dim,
        target_dim,
        linear_dim,
        checkpoint_every=1,
        patch_size=8,
        batch_size=3,
        num_target_views=1,
        num_input_views=6,
        resolution=512
    ):
        super().__init__()
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_target_views = num_target_views
        self.num_input_views = num_input_views
        self.patch_size = patch_size
        self.checkpoint_every = checkpoint_every

        self.input_linear = nn.Linear(input_dim, hidden, bias=False)
        self.target_linear = nn.Linear(target_dim, hidden, bias=False)

        self.norm_1 = nn.LayerNorm(hidden, bias=False)
        self.norm_2 = nn.LayerNorm(hidden, bias=False)

        self.model = nn.ModuleList()
        for idx in range(num_layers):
            block = Block(hidden=hidden, linear_dim=linear_dim)
            self.model.append(apply(block, init_weights, idx))

        self.output_linear = nn.Linear(hidden, patch_size * patch_size * 6, bias=False)
        self.output_xyz_linear = nn.Linear(hidden, patch_size * patch_size * 3, bias=False)
        self.xyz_net = xyzPredictor()

    def run_layers(self, start, end):
        def custom_forward(x):
            for i in range(start, min(end, len(self.model))):
                x = self.model[i](x)
            return x
        return custom_forward

    def forward(self, input_tokens, target_tokens):
        b = input_tokens.shape[0]
        num_input_tokens = input_tokens.shape[1]
        num_target_tokens = target_tokens.shape[1]

        input_tokens = self.input_linear(input_tokens)
        target_tokens = self.target_linear(target_tokens)

        x = self.norm_1(torch.cat([input_tokens, target_tokens], dim=1))

        for i in range(0, len(self.model), self.checkpoint_every):
            x = torch.utils.checkpoint.checkpoint(
                self.run_layers(i, i + self.checkpoint_every),
                x,
                use_reentrant=False
            )

        x = self.norm_2(x)

        output = self.output_linear(x)
        xyz_input = self.output_xyz_linear(x)

        # Reshape and predict xyz
        xyz_input = einops.rearrange(
            xyz_input,
            '(b) (o h w) (ph pw c) -> (b o) (h ph) (w pw) c',
            b=b,
            o=self.num_input_views + 1,
            h=self.resolution // self.patch_size,
            w=self.resolution // self.patch_size,
            ph=self.patch_size,
            pw=self.patch_size
        ).permute(0, 3, 1, 2)

        xyz = self.xyz_net(xyz_input).permute(0, 2, 3, 1)

        return output[:, :num_input_tokens], output[:, num_input_tokens:], xyz
