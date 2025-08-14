# -----------------------------------------------------------------------------
# 2025-08-13 (Release date)
# - Initial working PowerUp (DARE) Merge node implementation.
#
# 2025-08-14
# - Changed six boolean merge switches (input, middle, output blocks, attention,
#   convolution, normalization layers) to FLOAT sliders (0.0–2.0):
#     0.0 = no difference applied
#     1.0 = normal difference
#     >1.0 = amplified difference
# -----------------------------------------------------------------------------

import torch
import copy

class PowerUpDAREMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": ("MODEL",),
                "donor_model": ("MODEL",),
                "drop_rate": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "addition_multiplier": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 4.0, "step": 0.01}),
                "seed": ("INT", {"default": 99, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed"}),
                "merge_input_blocks": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "merge_middle_block": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "merge_output_blocks": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "merge_attention_layers": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "merge_convolution_layers": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "merge_normalization_layers": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = "model_merging"
    OUTPUT_IS_LIST = (False,)

    def merge(self, base_model, donor_model, seed, drop_rate, addition_multiplier,
              merge_input_blocks, merge_middle_block, merge_output_blocks,
              merge_attention_layers, merge_convolution_layers, merge_normalization_layers):

        device = next(base_model.model.parameters()).device
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

        base_sd = copy.deepcopy(base_model.model.state_dict())
        donor_sd = donor_model.model.state_dict()

        # Map of string key -> merge strength
        block_strengths = [
            ("input_blocks", merge_input_blocks),
            ("middle_block", merge_middle_block),
            ("output_blocks", merge_output_blocks),
            ("attn", merge_attention_layers),
            ("conv", merge_convolution_layers),
            ("Conv", merge_convolution_layers),
            ("norm", merge_normalization_layers),
        ]

        def merge_strength(name: str) -> float:
            for key_str, strength in block_strengths:
                if key_str in name:
                    return strength
            return 1.0

        for key in list(base_sd.keys()):
            if key in donor_sd:
                strength = merge_strength(key)
                if strength == 0.0:
                    continue

                b = base_sd[key].to(device)
                d = donor_sd[key].to(device)

                if b.shape != d.shape:
                    continue

                delta = d - b
                m = torch.empty_like(delta, device=device, dtype=delta.dtype).uniform_(0, 1, generator=gen) < drop_rate
                delta_tilde = delta * m.to(delta.dtype)
                denom = max(1e-8, 1.0 - float(drop_rate))
                delta_hat = delta_tilde / denom
                update = addition_multiplier * delta_hat * strength
                base_sd[key] = b + update

        merged_model = copy.deepcopy(base_model)
        merged_model.model.load_state_dict(base_sd)
        merged_model.model.to(device)

        return (merged_model,)


NODE_CLASS_MAPPINGS = {
    "PowerUp (DARE) Merge": PowerUpDAREMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PowerUp (DARE) Merge": "PowerUp (DARE) Merge"
}
