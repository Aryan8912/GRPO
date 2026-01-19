class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))

    def from_positions(self, position_ids):
        # Core RoPE block
        device = position_ids.device
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (
                self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
            inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos, sin

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (
                self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos_cos: Optional[torch.Tensor], pos_sin: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if pos_cos is None or pos_sin is None:
            return q, k

        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        if len(q_.shape) > len(pos_cos.shape):
            pos_cos = pos_cos.unsqueeze(1) # add head dim
            pos_sin = pos_sin.unsqueeze(1) # add head dim

        with torch.autocast(q.device.type, enabled=False):
            # we cache the KV after applied rope
            # query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
            # pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            # pos_sin = pos_sin.type_as(q_)
            # pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, q_)
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)
