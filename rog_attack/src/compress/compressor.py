import numpy as np
import math
import torch

class UniformQuantizer:
    def __init__(self, config):
        self.quantbound = config.quant_level - 1

    def compress(self, arr):
        """
        quantize a given arr array with unifrom quant.
        """
        max_val = torch.max(arr.abs())
        sign_arr = arr.sign()
        quantized_arr = (arr/max_val)*self.quantbound
        quantized_arr = torch.abs(quantized_arr)
        quantized_arr = torch.round(quantized_arr).to(torch.int)
        
        quantized_set = dict(max_val=max_val, signs=sign_arr, quantized_arr=quantized_arr)
        return quantized_set
    
    def decompress(self, quantized_set):
        """
        dequantize a given array which is uniformed quantized.
        """
        coefficients = quantized_set["max_val"]/self.quantbound  * quantized_set["signs"] 
        dequant_arr =  coefficients * quantized_set["quantized_arr"]

        return dequant_arr


class SignSGDCompressor:
    def __init__(self, config):
        pass 

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        encoded_tensor = (tensor >= 0).to(torch.float)
        return encoded_tensor

    def decompress(self, tensor):
        """Decode the signs to float format """
        decoded_tensor = tensor * 2 - 1
        return decoded_tensor

class Topk:

    def __init__(self, config):
        self.sparsity = config.sparsity

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        k = np.ceil(tensor.numel()*(1-self.sparsity)).astype(int)        
        top_k_element, top_k_index = torch.kthvalue(-tensor.abs().flatten(), k)
        tensor_masked = (tensor.abs() > -top_k_element) * tensor

        return tensor_masked

    def decompress(self, tensor):
        """Return the original tensor"""
        return tensor


class QsgdQuantizer:
    def __init__(self, config):
        self.quantlevel = config.quant_level 
        self.quantbound = config.quant_level - 1

    def compress(self, arr):
        # norm = arr.norm()
        norm = torch.max(arr.abs())
        abs_arr = arr.abs()

        level_float = abs_arr / norm * self.quantbound 
        lower_level = level_float.floor()
        rand_variable = torch.empty_like(arr).uniform_() 
        is_upper_level = rand_variable < (level_float - lower_level)
        new_level = (lower_level + is_upper_level)
        quantized_arr = torch.round(new_level)

        sign = arr.sign()
        quantized_set = dict(norm=norm, signs=sign, quantized_arr=quantized_arr)

        return quantized_set

    def decompress(self, quantized_set):
        coefficients = quantized_set["norm"]/self.quantbound * quantized_set["signs"]
        dequant_arr = coefficients * quantized_set["quantized_arr"]

        return dequant_arr
    


class DPSGDCompressor:
    """
    Per-parameter DP-SGD style clipping + noise for a single gradient pass.

    - Clips each parameter tensor to L2 norm <= dp_clip.
    - Adds Gaussian noise with std = dp_sigma * dp_clip to that tensor.
    - Returns the final noisy gradient directly (no metadata).
    """
    def __init__(self, config):
        self.dp_clip  = getattr(config, "dp_clip", 1.0)     # C
        self.dp_sigma = getattr(config, "dp_sigma", 0.0)    # sigma
        seed = getattr(config, "dp_seed", None)

        # Optional deterministic noise
        self._gen = None
        if seed is not None:
            # Use the same device as noise destination (set per call)
            self._gen = torch.Generator()
            self._gen.manual_seed(int(seed))

    def compress(self, tensor, **kwargs):
        # L2 clip per parameter tensor (per-layer clipping)
        
        l2 = tensor.norm(2)
        clip_factor = min(1.0, float(self.dp_clip) / (l2.item() + 1e-12))
        clipped = tensor * clip_factor

        # Add Gaussian noise (single-shot, not batch-averaged)
        if self.dp_sigma > 0.0:
            std = float(self.dp_sigma) * float(self.dp_clip)
            noise = torch.normal(
                mean=0.0,
                std=std,
                size=clipped.shape,
                device=clipped.device,
                dtype=clipped.dtype,
                generator=self._gen
            )
            clipped = clipped + noise

        # Return final gradient directly
        return clipped

    def decompress(self, tensor):
        # Identity (already noisy gradient)
        return tensor
    

class PruneLargest:
    """
    Zero out the largest-magnitude gradients by fraction `pruning_rate`.

    Modes:
      - per_tensor (default): compute threshold per tensor.
      - global: compute one threshold across all tensors via `prefit(grads)`.

    Config keys:
      - pruning_rate: float in (0,1), e.g., 0.3
      - pruning_mode: "per_tensor" or "global"
    """
    def __init__(self, config):
        self.pruning_rate = float(getattr(config, "pruning_rate", 0.3))
        assert 0.0 < self.pruning_rate < 1.0, "pruning_rate must be in (0,1)."
        self.mode = getattr(config, "pruning_mode", "per_tensor").lower()
        self._global_threshold = None  # set by prefit in global mode

    @property
    def requires_prefit(self):
        return self.mode == "global"

    @torch.no_grad()
    def prefit(self, grads_list):
        """
        Compute global threshold across all tensors (skip 0-dim scalars).
        Call this once per gradient step if pruning_mode == 'global'.
        """
        if self.mode != "global":
            return

        # Collect absolute values across all tensors
        abs_flat_parts = []
        for g in grads_list:
            if g is None:
                continue
            if g.ndim == 0:
                # skip scalars from threshold calc
                continue
            abs_flat_parts.append(g.detach().abs().flatten())

        if not abs_flat_parts:
            self._global_threshold = None
            return

        abs_all = torch.cat(abs_flat_parts, dim=0)
        n = abs_all.numel()
        # number to prune globally
        k_prune = max(1, int(math.ceil(n * self.pruning_rate)))
        # threshold is the (n - k_prune + 1)-th smallest => keeps <= threshold, zeros > threshold
        kth_index = n - k_prune + 1
        # torch.kthvalue is 1-indexed for k
        thresh_val = torch.kthvalue(abs_all, kth_index).values
        # Store as Python float to avoid device issues
        self._global_threshold = float(thresh_val.item())

    @torch.no_grad()
    def compress(self, tensor, **kwargs):
        """
        Return pruned tensor (largest magnitudes set to zero).
        """
        if tensor is None or tensor.ndim == 0:
            # Keep scalars as-is (consistent with your reference implementation)
            return tensor

        abs_flat = tensor.abs().flatten()
        if self.mode == "global":
            if self._global_threshold is None:
                raise RuntimeError("PruneLargest in 'global' mode requires prefit(grads) before compress().")
            threshold = self._global_threshold
            # keep values <= threshold
            mask_flat = (abs_flat <= threshold)
        else:
            # per-tensor threshold
            n = abs_flat.numel()
            k_prune = max(1, int(math.ceil(n * self.pruning_rate)))
            kth_index = n - k_prune + 1
            thresh_val = torch.kthvalue(abs_flat, kth_index).values
            mask_flat = (abs_flat <= thresh_val)

        mask = mask_flat.view_as(tensor)
        pruned = tensor * mask
        return pruned

    def decompress(self, tensor):
        # Identity — already pruned
        return tensor


class ErisCompressor:
    """
    ERIS compressor for precomputed aggregator masks.

    Workflow:
      1) prefit(grads_list, masks, aggregator_id, k, seed=None):
         - masks: list of int tensors/ndarrays; same shapes as grads; each entry is in [0, n_aggregators-1]
         - aggregator_id: which shard/aggregator to expose (keep)
         - k: number of coords to keep globally among the *selected* coords (random-k)
         - builds a boolean keep-mask per tensor and a global scaling factor
      2) compress(tensor): applies the keep-mask for this tensor (cursor-based) and scales kept entries
      3) decompress(tensor): identity

    Config (optional):
      - eris_k (int): default k if not passed to prefit
      - eris_unbiased (bool): if True (default), scale kept entries by d_sel / k
    """
    def __init__(self, default_k, config):
        self.default_k = default_k
        # self.unbiased  = bool(getattr(config, "eris_unbiased", True))

        # state set by prefit
        self._keep_masks = None   # List[torch.BoolTensor], CPU
        self._scale = 1.0
        self._cursor = 0

    @property
    def requires_prefit(self):
        return True

    @torch.no_grad()
    def prefit(self, grads_list, masks, aggregator_id, k=None, seed=None):
        """
        Build per-tensor keep masks and global scale.

        grads_list: List[torch.Tensor]
        masks:      List[np.ndarray or torch.Tensor] with same shapes as grads_list;
                    entries are int aggregator ids.
        aggregator_id: int
        k: int or None (falls back to self.default_k)
        seed: optional int for reproducibility
        """
        assert len(grads_list) == len(masks), "grads_list and masks must have same length"

        # Convert masks -> torch on CPU; build 'selected' mask per tensor
        selected_masks = []
        counts = []
        for g, m in zip(grads_list, masks):
            if g is None:
                selected_masks.append(None)
                counts.append(0)
                continue
            if g.ndim == 0:
                # Scalar params: do not include in selection (keep zero)
                sel = torch.zeros((), dtype=torch.bool)
                selected_masks.append(sel)
                counts.append(0)
                continue
            mt = torch.as_tensor(m, device="cpu")
            sel = (mt == int(aggregator_id)).to(torch.bool)  # CPU bool
            assert tuple(sel.shape) == tuple(g.shape), "Mask shape must match grad shape"
            selected_masks.append(sel)
            counts.append(int(sel.sum().item()))

        d_sel = int(sum(counts))
        self._keep_masks = []
        self._scale = 1.0

        if d_sel == 0:
            # Nothing selected: keep-masks are all zeros
            for sel in selected_masks:
                if sel is None:
                    self._keep_masks.append(None)
                elif sel.ndim == 0:
                    self._keep_masks.append(torch.zeros((), dtype=torch.bool))
                else:
                    self._keep_masks.append(torch.zeros_like(sel, dtype=torch.bool))
            self._scale = 1.0
            return

        # Determine k to keep among selected coordinates
        k_eff = int(k if (k is not None) else self.default_k)
        if k_eff <= 0 or k_eff >= d_sel:
            # Keep all selected coordinates; no scaling needed (or scaling=1)
            for sel in selected_masks:
                if sel is None:
                    self._keep_masks.append(None)
                elif sel.ndim == 0:
                    self._keep_masks.append(torch.zeros((), dtype=torch.bool))
                else:
                    self._keep_masks.append(sel.clone())  # keep all selected
            self._scale = 1.0
            return

        # Choose k indices uniformly among the d_sel selected positions
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))
        # indices in [0, d_sel)
        chosen = torch.randperm(d_sel, generator=gen)[:k_eff]
        chosen = chosen.sort().values  # sorted for stable slicing

        # Build a flat boolean 'keep' of length d_sel with True at chosen positions
        keep_all = torch.zeros(d_sel, dtype=torch.bool)
        keep_all[chosen] = True

        # Map keep_all back to per-tensor boolean masks
        offset = 0
        for sel, cnt in zip(selected_masks, counts):
            if sel is None:
                self._keep_masks.append(None)
                continue
            if sel.ndim == 0:
                self._keep_masks.append(torch.zeros((), dtype=torch.bool))
                continue
            if cnt == 0:
                self._keep_masks.append(torch.zeros_like(sel, dtype=torch.bool))
                continue

            # keep segment for this tensor among its selected coords
            segment = keep_all[offset:offset + cnt]
            offset += cnt

            # expand segment into full-shape mask: put segment where sel==True
            sel_flat = sel.view(-1)
            km_flat = torch.zeros_like(sel_flat, dtype=torch.bool)
            km_flat[sel_flat] = segment  # assign chosen positions
            self._keep_masks.append(km_flat.view_as(sel))

        # Unbiased scaling d_sel / k_eff (optional)
        # self._scale = (float(d_sel) / float(k_eff)) if self.unbiased else 1.0
        self.scale = 1.0 

    @torch.no_grad()
    def compress(self, tensor, **kwargs):
        """
        Apply keep-mask for the current tensor (cursor-based) and scale kept entries.
        """
        if tensor is None:
            return None
        if self._keep_masks is None:
            raise RuntimeError("ErisCompressor requires prefit(...) before compress().")

        if self._cursor >= len(self._keep_masks):
            raise RuntimeError("ErisCompressor cursor overflow — did you call compress more times than prefit masks?")

        km = self._keep_masks[self._cursor]
        self._cursor += 1

        if km is None:
            return tensor  # unusual; but keep as-is

        if tensor.ndim == 0:
            # scalars: forced to zero
            return torch.zeros_like(tensor)

        # Move mask to the same device as tensor without copying data back
        km_dev = km.to(device=tensor.device, dtype=torch.bool, non_blocking=True)
        # zero others, scale kept coords
        out = torch.where(km_dev, tensor * self._scale, torch.zeros_like(tensor))
        return out

    def decompress(self, tensor):
        # Identity — already sparse/scaled
        return tensor


class ErisPartialCompressor:
    """
    Partial-split ERIS (mask-free):
      - Select exactly k ≈ d / n_splits coordinates *globally* at random.
      - Zero everything else.
      - No scaling (matches your DLG/iDLG partial split behavior).

    Config:
      - n_aggregators (int): number of splits
      - eris_partial_unbiased (bool, default False): if True, scale kept entries by d/k
    """
    def __init__(self, config):
        self.n_splits = int(getattr(config, "n_aggregators", 2))
        # self.unbiased = bool(getattr(config, "eris_partial_unbiased", False))
        self.unbiased = False

        # state after prefit
        self._keep_masks = None  # List[torch.BoolTensor] on CPU
        self._scale = 1.0
        self._cursor = 0

    @property
    def requires_prefit(self):
        return True

    @torch.no_grad()
    def prefit(self, grads_list, n_splits=None, k=None, seed=None):
        """
        Build per-tensor boolean keep masks by global random selection.

        grads_list : List[torch.Tensor]
        n_splits   : optional override for number of splits
        k          : optional explicit number of kept coords; if None, k = floor(d / n_splits)
        seed       : RNG seed for reproducibility
        """
        ns = int(n_splits) if n_splits is not None else self.n_splits

        # 1) Gather sizes & total dimension d (skip scalars)
        sizes = []
        for g in grads_list:
            if g is None:
                sizes.append(0)
            elif g.ndim == 0:
                sizes.append(0)
            else:
                sizes.append(g.numel())
        d = int(sum(sizes))

        # 2) Decide k
        if k is None:
            if ns <= 0:
                raise ValueError("n_splits must be >= 1.")
            k_eff = max(1, d // ns)
        else:
            k_eff = max(1, int(k))
        k_eff = min(k_eff, d)  # safety

        # 3) Sample k indices globally
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))
        if d == 0:
            # edge case: nothing to keep
            self._keep_masks = []
            self._scale = 1.0
            self._cursor = 0
            return

        perm = torch.randperm(d, generator=gen)
        keep_global = torch.zeros(d, dtype=torch.bool)
        keep_global[perm[:k_eff]] = True

        # 4) Map back to per-tensor masks
        self._keep_masks = []
        self._cursor = 0
        offset = 0
        for g, sz in zip(grads_list, sizes):
            if g is None:
                self._keep_masks.append(None)
                continue
            if g.ndim == 0 or sz == 0:
                self._keep_masks.append(torch.zeros((), dtype=torch.bool))
                continue
            segment = keep_global[offset:offset + sz]
            offset += sz
            self._keep_masks.append(segment.view_as(g).cpu())

        # 5) Scaling (disabled by default to match your earlier partial-split)
        self._scale = (float(d) / float(k_eff)) if self.unbiased and k_eff > 0 else 1.0

    @torch.no_grad()
    def compress(self, tensor, **kwargs):
        if tensor is None:
            return None
        if self._keep_masks is None:
            raise RuntimeError("ErisPartialCompressor requires prefit(...) before compress().")
        if self._cursor >= len(self._keep_masks):
            raise RuntimeError("Cursor overflow in ErisPartialCompressor.compress().")

        km = self._keep_masks[self._cursor]
        self._cursor += 1

        if km is None:
            return tensor
        if tensor.ndim == 0:
            return torch.zeros_like(tensor)

        km_dev = km.to(device=tensor.device, dtype=torch.bool, non_blocking=True)
        # return torch.where(km_dev, tensor * self._scale, torch.zeros_like(tensor))
        out = torch.where(km_dev, tensor * self._scale, torch.zeros_like(tensor))
        # count the number of kept coordinates
        n_keept = int(km_dev.sum().item()) 
        return out, n_keept

    def decompress(self, tensor):
        return tensor

class RandomCompressor:
    def __init__(self, config):
        self.config = config

    def compress(self, arr):
        """
        give a random array with the same shape as arr.
        """
        return torch.randn_like(arr) * 0.002

    def decompress(self, arr):
        return arr
