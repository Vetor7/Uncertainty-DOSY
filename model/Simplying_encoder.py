import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import math

class Simply_encode(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, Dropout, mult_h=20, norm_type="ln", layer_idx=0):
        super().__init__()

        inner_dim = mult_h * hidden_size
        self.layer_idx = layer_idx

        norm_layer = {
            "ln": lambda: nn.LayerNorm(hidden_size, eps=1e-6),
            "none": lambda: nn.Identity() if layer_idx > 0 else nn.LayerNorm(hidden_size, eps=1e-6)
        }.get(norm_type, lambda: None)()
        self.ln_1 = norm_layer
        self.attn = Simplify_Attention(hidden_size, num_attention_heads, Dropout, layer_idx=layer_idx)
        self.mlp = MLP(inner_dim, hidden_size, Dropout)
        self.attn_block_resid_gain = nn.Parameter(torch.ones(1), requires_grad=True)
        self.mlp_block_resid_gain = nn.Parameter(torch.full((1,), 0.1), requires_grad=True)

    def forward(self, hidden_states: torch.FloatTensor, attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(hidden_states, attention_mask=attention_mask, head_mask=head_mask)
        feed_forward_hidden_states = self.mlp(hidden_states)
        outputs = self.mlp_block_resid_gain * feed_forward_hidden_states + self.attn_block_resid_gain * attn_outputs
        return outputs

class MLP(nn.Module):
    def __init__(self, intermediate_size, hidden_size, Dropout=0):
        super().__init__()
        embed_dim = hidden_size

        self.c_fc = MyConv1D(intermediate_size, embed_dim, bias=False)
        self.c_proj = MyConv1D(embed_dim, intermediate_size, bias=False)

        self.act = LeakyReLU(negative_slope=0)

        self.dropout = nn.Dropout(Dropout)

    def forward(
            self, hidden_states: Optional[Tuple[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class MyConv1D(nn.Module):
    def __init__(
            self,
            nf,
            nx,
            resid_gain=None,
            skip_gain=None,
            trainable_gains=False,
            init_type="normal",
            bias=True,
    ):
        super().__init__()
        self.nf = nf

        if bias:
            self.bias = nn.Parameter(torch.zeros(nf))
        else:
            self.bias = nn.Parameter(torch.zeros(nf), requires_grad=False)

        if skip_gain is None:
            # Standard linear layer
            self.weight = nn.Parameter(torch.empty(nx, nf))
            if init_type == "orth":
                nn.init.orthogonal_(self.weight)
            elif init_type == "id":
                self.weight.data = torch.eye(nx)
            elif init_type == "normal":
                nn.init.normal_(self.weight, std=0.02)
            else:
                raise NotImplementedError
            self.skip = False

        elif skip_gain is not None:
            # Reparameterised linear layer
            assert nx == nf
            self.resid_gain = nn.Parameter(torch.Tensor([resid_gain]), requires_grad=trainable_gains)
            self.skip_gain = nn.Parameter(torch.Tensor([skip_gain]),requires_grad=trainable_gains)

            self.weight = nn.Parameter(torch.zeros(nx, nx))
            if init_type == "orth":
                self.id = nn.init.orthogonal_(torch.empty(nx, nx))
            elif init_type == "id":
                self.id = torch.eye(nx)
            elif init_type == "normal":
                self.id = nn.init.normal_(
                    torch.empty(nx, nx), std=1 / math.sqrt(nx)
                )
            else:
                raise NotImplementedError
            self.skip = True
            self.init_type = init_type

    def forward(self, x):
        device = x.device
        size_out = x.size()[:-1] + (self.nf,)
        if self.skip:
            if self.resid_gain == 0 and self.init_type == "id":
                x = torch.add(self.bias, x * self.skip_gain).to(device)
            else:
                x = torch.addmm(
                    self.bias,
                    x.view(-1, x.size(-1)),
                    self.resid_gain * self.weight + self.skip_gain * self.id.to(device),
                ).to(device)
        else:
            x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight).to(device)

        x = x.view(size_out)

        return x

class LeakyReLU(nn.Module):
    # LeakyReLU nonlinearity.
    __constants__ = ["inplace", "negative_slope"]
    inplace: bool
    negative_slope: float

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.where(input >= 0.0, input, input * self.negative_slope)

    def extra_repr(self) -> str:
        inplace_str = ", inplace=True" if self.inplace else ""
        return "negative_slope={}{}".format(self.negative_slope, inplace_str)

class Simplify_Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, Dropout, is_cross_attention=False, layer_idx=None,
                 max_positions=3000):
        super().__init__()
        assert is_cross_attention == False

        self.centre = True
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.layer_idx = layer_idx
        self.qk_attn = MyConv1D(
            2 * self.embed_dim,
            self.embed_dim,
        )
        self.c_proj = nn.Identity()
        self.split_size = self.embed_dim
        query_weight, key_weight = self.qk_attn.weight.data.split(
            self.split_size, dim=1
        )
        query_weight.normal_(mean=0.0, std=0)
        self.attn_dropout = nn.Dropout(Dropout)
        self.resid_dropout = nn.Dropout(Dropout)

        self.attn_mat_resid_gain = nn.Parameter(
            torch.ones((1, self.num_heads, 1, 1)),
            requires_grad=True,
        )
        self.attn_mat_skip_gain = nn.Parameter(
            torch.ones((1, self.num_heads, 1, 1)),
            requires_grad=True,
        )
        self.centre_attn_gain = True
        self.centre_attn_gain = nn.Parameter(
            torch.ones((1, self.num_heads, 1, 1)),
            requires_grad=True
        )

        if (layer_idx == 0):
            self.v_attn = MyConv1D(
                self.embed_dim,
                self.embed_dim,
                resid_gain=1,
                skip_gain=1,
                trainable_gains=True,
                init_type="id",
                bias=False,
            )
        else:
            self.v_attn = nn.Identity()
        self.pruned_heads = set()
        uniform_causal_attn_mat = torch.ones(
            (max_positions, max_positions), dtype=torch.float32
        ) / torch.arange(1, max_positions + 1).view(-1, 1)
        self.register_buffer(
            "uniform_causal_attn_mat",
            torch.tril(
                uniform_causal_attn_mat,
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        self.register_buffer(
            "diag",
            torch.eye(max_positions).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights = attn_weights / torch.full(
            [],
            value.size(-1) ** 0.5,
            dtype=attn_weights.dtype,
            device=attn_weights.device,
        )
        query_length, key_length = query.size(-2), key.size(-2)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        new_attn_weights = self.attn_mat_resid_gain * attn_weights.type(value.dtype)
        if self.centre:
            post_sm_bias_matrix = (
                                    self.attn_mat_skip_gain * self.diag[:, :, :key_length, :key_length]
                                  ) - self.centre_attn_gain * (
                                      self.uniform_causal_attn_mat[
                                      :, :, key_length - query_length: key_length, :key_length
                                      ]
                                  )
            new_attn_weights = new_attn_weights + post_sm_bias_matrix

        new_attn_weights = self.attn_dropout(new_attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            new_attn_weights = new_attn_weights * head_mask

        attn_output = torch.matmul(new_attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        (query, key) = self.qk_attn(hidden_states).split(self.split_size, dim=2)
        value = self.v_attn(hidden_states)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        proj_output = self.c_proj(attn_output)
        proj_output = self.resid_dropout(proj_output)
        outputs = proj_output
        return outputs
