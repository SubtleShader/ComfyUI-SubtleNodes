# -----------------------------------------------------------------------------
# 2025-08-13 (Release date)
# - Initial working DARE Merge node implementation.
#
# 2025-08-14
# - Changed six boolean merge switches (input, middle, output blocks, attention,
#   convolution, normalization layers) to FLOAT sliders (0.0-2.0):
#     0.0 = no difference applied
#     1.0 = normal difference
#     >1.0 = amplified difference
#
# 2026-02-11
# - Added automatic LoRA/LoKR baking for donor model before merge
# - Use ComfyUI patch system with correct ("diff", (delta,)) format
# -----------------------------------------------------------------------------

import torch
import comfy.lora

class DAREMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": ("MODEL",),
                "donor_model": ("MODEL",),
                "drop_rate": ("FLOAT", {
                    "default": 0.55,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "addition_multiplier": ("FLOAT", {
                    "default": 0.40,
                    "min": -1.0,
                    "max": 4.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "seed": ("INT", {
                    "default": 99,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "merge_input_blocks": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
                "merge_middle_block": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
                "merge_output_blocks": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
                "merge_attention_layers": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
                "merge_convolution_layers": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
                "merge_normalization_layers": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "merge"
    CATEGORY = "SubtleNodes"
    OUTPUT_IS_LIST = (False,)

    def merge(self, base_model, donor_model, seed, drop_rate, addition_multiplier,
              merge_input_blocks, merge_middle_block, merge_output_blocks,
              merge_attention_layers, merge_convolution_layers, merge_normalization_layers):

        print(f"\n{'='*60}")
        print(f"DARE Merge")
        print(f"{'='*60}")

        device = next(base_model.model.parameters()).device
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

        def merge_strength(name: str) -> float:
            strength = 1.0
            if "input_blocks" in name:
                strength *= merge_input_blocks
            elif "middle_block" in name:
                strength *= merge_middle_block
            elif "output_blocks" in name:
                strength *= merge_output_blocks
            elif "double_blocks" in name:
                strength *= merge_middle_block
            elif "single_blocks" in name:
                strength *= merge_output_blocks
            elif "img_in" in name or "txt_in" in name or "time_in" in name:
                strength *= merge_input_blocks
            elif "final_layer" in name:
                strength *= merge_output_blocks
            
            if "attn" in name or "attention" in name:
                strength *= merge_attention_layers
            elif "conv" in name or "Conv" in name:
                strength *= merge_convolution_layers
            elif "norm" in name or "modulation" in name:
                strength *= merge_normalization_layers
            elif "mlp" in name or "linear" in name:
                strength *= merge_attention_layers
            return strength

        # Clone base model
        merged_model = base_model.clone()
        
        # Get state dicts
        base_sd = base_model.model.state_dict()
        donor_sd_orig = donor_model.model.state_dict()
        
        # Bake LoRA/LoKR into donor if present
        has_patches = len(donor_model.patches) > 0
        if has_patches:
            print(f"Baking {len(donor_model.patches)} patches...")
            donor_sd = {}
            for key in donor_sd_orig.keys():
                weight = donor_sd_orig[key].clone().to(device)
                if key in donor_model.patches:
                    patches = donor_model.patches[key]
                    weight = comfy.lora.calculate_weight(patches, weight, key, intermediate_dtype=weight.dtype)
                donor_sd[key] = weight
        else:
            donor_sd = donor_sd_orig

        # Calculate DARE deltas and add them as patches
        patches_to_add = {}
        merged_count = 0
        
        for key in list(base_sd.keys()):
            if key in donor_sd:
                strength = merge_strength(key)
                if strength == 0.0:
                    continue

                b = base_sd[key].to(device)
                d = donor_sd[key].to(device)

                if b.shape != d.shape:
                    continue

                # Calculate DARE delta
                delta = d - b
                m = torch.empty_like(delta, device=device, dtype=delta.dtype).uniform_(0, 1, generator=gen) >= drop_rate
                delta_tilde = delta * m.to(delta.dtype)
                denom = max(1e-8, 1.0 - float(drop_rate))
                delta_hat = delta_tilde / denom
                update = addition_multiplier * delta_hat * strength
                
                # Format as ComfyUI patch: ("diff", (weight_diff,))
                # The strength will be applied by add_patches
                patches_to_add[key] = ("diff", (update,))
                merged_count += 1

        # Add all DARE patches at once
        # strength_patch=1.0 means use patches at full strength
        # strength_model=1.0 means keep base model at full strength
        if patches_to_add:
            merged_model.add_patches(patches_to_add, strength_model=1.0, strength_patch=1.0)
        
        print(f"Added DARE patches to {merged_count} layers")
        print(f"{'='*60}\n")

        return (merged_model,)


NODE_CLASS_MAPPINGS = {
    "DARE Merge": DAREMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DARE Merge": "DARE Merge"
}
