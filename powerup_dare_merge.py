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
                "merge_input_blocks": ("BOOLEAN", {"default": True}),
                "merge_middle_block": ("BOOLEAN", {"default": True}),
                "merge_output_blocks": ("BOOLEAN", {"default": True}),
                "merge_attention_layers": ("BOOLEAN", {"default": True}),
                "merge_convolution_layers": ("BOOLEAN", {"default": True}),
                "merge_normalization_layers": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 99, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"
    CATEGORY = "model_merging"
    OUTPUT_IS_LIST = (False,)

    def merge(self, base_model, donor_model, drop_rate, addition_multiplier,
              merge_input_blocks, merge_middle_block, merge_output_blocks,
              merge_attention_layers, merge_convolution_layers, merge_normalization_layers,
              seed):

        device = next(base_model.model.parameters()).device
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

        base_sd = copy.deepcopy(base_model.model.state_dict())
        donor_sd = donor_model.model.state_dict()

        def should_merge_layer(name: str) -> bool:
            if not merge_input_blocks and "input_blocks" in name:
                return False
            if not merge_middle_block and "middle_block" in name:
                return False
            if not merge_output_blocks and "output_blocks" in name:
                return False
            if not merge_attention_layers and "attn" in name:
                return False
            if not merge_convolution_layers and any(x in name for x in ["conv", "Conv"]):
                return False
            if not merge_normalization_layers and "norm" in name:
                return False
            return True

        for key in list(base_sd.keys()):
            if key in donor_sd and should_merge_layer(key):
                b = base_sd[key].to(device)
                d = donor_sd[key].to(device)

                if b.shape != d.shape:
                    continue

                delta = d - b
                m = torch.empty_like(delta, device=device, dtype=delta.dtype).uniform_(0, 1, generator=gen) < drop_rate
                delta_tilde = delta * m.to(delta.dtype)
                denom = max(1e-8, 1.0 - float(drop_rate))
                delta_hat = delta_tilde / denom
                update = addition_multiplier * delta_hat
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
