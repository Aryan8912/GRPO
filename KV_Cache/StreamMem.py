import torch
import torch.nn.functional as F


class StreamingKVCache:
    def __init__(self, num_layers, num_heads, head_dim, max_cache_size):
        """
        Args:
            num_layers: transformer layers
            num_heads: attention heads
            head_dim: dimension per head
            max_cache_size (M): total KV capacity
        """
        self.L = num_layers
        self.H = num_heads
        self.D = head_dim
        self.M = max_cache_size

        # Global cache
        self.K = [torch.empty(0, self.H, self.D) for _ in range(self.L)]
        self.V = [torch.empty(0, self.H, self.D) for _ in range(self.L)]

        # Score matrix (one row per layer)
        self.s = [torch.empty(0) for _ in range(self.L)]

    # ---------------------------------------------------
    # Frame similarity filter
    # ---------------------------------------------------
    def filter_frames(self, frames, threshold=0.9):
        """
        Remove highly similar consecutive frames.
        """
        if len(frames) <= 1:
            return frames

        filtered = [frames[0]]
        for i in range(1, len(frames)):
            sim = F.cosine_similarity(
                frames[i].flatten(),
                frames[i - 1].flatten(),
                dim=0
            )
            if sim < threshold:
                filtered.append(frames[i])

        return torch.stack(filtered)

    # ---------------------------------------------------
    # Encode frames -> KV + scores
    # ---------------------------------------------------
    def encode(self, frames, template_tokens):
        """
        Simulated encoder.
        Replace with actual transformer forward pass.
        """
        B = frames.shape[0]

        Ki = []
        Vi = []
        si = []

        for l in range(self.L):
            k = torch.randn(B, self.H, self.D)
            v = torch.randn(B, self.H, self.D)

            # Example score: mean attention magnitude
            score = torch.norm(k, dim=-1).mean(dim=-1)

            Ki.append(k)
            Vi.append(v)
            si.append(score)

        return Ki, Vi, si

    # ---------------------------------------------------
    # Top-K pruning
    # ---------------------------------------------------
    def topk_prune(self):
        for l in range(self.L):
            if self.K[l].shape[0] > self.M:
                scores = self.s[l]
                topk_indices = torch.topk(scores, k=self.M).indices

                self.K[l] = self.K[l][topk_indices]
                self.V[l] = self.V[l][topk_indices]
                self.s[l] = self.s[l][topk_indices]

    # ---------------------------------------------------
    # Merge compression
    # ---------------------------------------------------
    def merge(self, tensor):
        """
        Merge consecutive tokens by averaging.
        """
        if tensor.shape[0] < 2:
            return tensor

        even = tensor[0::2]
        odd = tensor[1::2]

        min_len = min(even.shape[0], odd.shape[0])
        merged = (even[:min_len] + odd[:min_len]) / 2.0

        return merged

    # ---------------------------------------------------
    # Main streaming loop step
    # ---------------------------------------------------
    def process_batch(self, frames, template_tokens):
        """
        Implements your algorithm.
        """

        # v'i = Filter(vi)
        filtered_frames = self.filter_frames(frames)

        # Ki, Vi, si = Encode(v'i, Q)
        Ki, Vi, si = self.encode(filtered_frames, template_tokens)

        # if |K| > M -> TopK prune
        self.topk_prune()

        # Append Ki, Vi, si to global cache  (Equation 1)
        for l in range(self.L):
            self.K[l] = torch.cat([self.K[l], Ki[l]], dim=0)
            self.V[l] = torch.cat([self.V[l], Vi[l]], dim=0)
            self.s[l] = torch.cat([self.s[l], si[l]], dim=0)

        # Insert Merge(Ki), Merge(Vi) (Equation 2)
        for l in range(self.L):
            merged_K = self.merge(Ki[l])
            merged_V = self.merge(Vi[l])

            self.K[l] = torch.cat([self.K[l], merged_K], dim=0)
            self.V[l] = torch.cat([self.V[l], merged_V], dim=0)

        # Final prune again to maintain budget
        self.topk_prune()

    # ---------------------------------------------------
    def cache_size(self):
        return [self.K[l].shape[0] for l in range(self.L)]