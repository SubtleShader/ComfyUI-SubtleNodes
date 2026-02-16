"""Subtle Merge Node - Merges two checkpoints using importance-weighted merging."""


# -----------------------------------------------------------------------------
# 2026-02-16 (Release date)
# - Initial working node implementation.
# - Added automatic LoRA/LoKR baking for donor model before merge
# -----------------------------------------------------------------------------

import torch
import gc
import comfy.lora
from typing import Tuple


class SubtleMerge:
    """Merge two checkpoints using importance-weighted merging.
    
    Computes importance scores based on weight magnitudes and merges accordingly.
    Automatically bakes LoRA/LoKR patches before merging.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": ("MODEL",),
                "donor_model": ("MODEL",),
                "donor_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "donor_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": -1.0,
                    "max": 5.0,
                    "step": 0.05,
                    "display": "slider"
                }),
            },
            "optional": {
                "importance_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "device": (["auto", "cpu"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("merged_model",)
    FUNCTION = "merge_checkpoints"
    CATEGORY = "SubtleNodes"
    
    def get_device_settings(self, device_preset: str) -> tuple[str, int, int]:
        """Get device, batch size, and cleanup frequency based on preset.
        
        Auto mode automatically detects VRAM and picks optimal settings.
        
        Args:
            device_preset: "auto" (detects VRAM automatically) or "cpu" (force CPU)
            
        Returns:
            Tuple of (device_str, batch_size, cleanup_frequency)
        """
        if device_preset == "cpu":
            return "cpu", 0, 5
        
        # Auto mode - detect VRAM and configure automatically
        if torch.cuda.is_available():
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                if vram_gb >= 22:
                    return "cuda", 20, 50
                elif vram_gb >= 14:
                    return "cuda", 10, 20
                elif vram_gb >= 10:
                    return "cuda", 5, 10
                else:
                    return "cuda", 2, 5
            except Exception as e:
                print(f"Auto-detection failed ({e}), using conservative GPU settings")
                return "cuda", 5, 10
        else:
            return "cpu", 0, 5
    
    def importance_merge(
        self,
        weight_a: torch.Tensor,
        weight_b: torch.Tensor,
        donor_ratio: float,
        donor_factor: float,
        importance_threshold: float,
        device: str = "cpu"
    ) -> torch.Tensor:
        """Merge two weight tensors using importance-weighted merging.
        
        Args:
            weight_a: Base model weights
            weight_b: Donor model weights
            ratio: Base interpolation factor (0 = all A, 1 = all B)
            donor_factor: Importance amplification (1.0 = full, 0.0 = simple lerp)
            importance_threshold: Minimum importance to apply donor weighting
            device: Computation device ("cpu" or "cuda")
            
        Returns:
            Merged weights
        """
        original_dtype = weight_a.dtype
        
        wa = weight_a.to(device=device, dtype=torch.float32)
        wb = weight_b.to(device=device, dtype=torch.float32)
        
        if donor_factor < 0:
            # Passthrough mode
            result = wa.to(dtype=original_dtype)
            del wa, wb
            return result
        
        if donor_factor == 0:
            # Simple linear interpolation
            merged = wa.lerp(wb, donor_ratio)
            result = merged.to(dtype=original_dtype)
            del wa, wb, merged
            return result
        
        # Compute delta and importance
        delta = wb - wa
        importance = torch.abs(delta) / (torch.abs(wa) + 1e-8)
        del delta
        
        # Normalize to [0, 1] range per-tensor
        max_importance = importance.max().item()
        if max_importance > 0:
            importance.div_(max_importance)
        
        # Apply threshold
        importance_mask = importance > importance_threshold
        
        # Compute donor weight
        donor_weight = donor_ratio * (1.0 + donor_factor * importance)
        donor_weight.clamp_(0.0, 1.0)
        
        # Apply merge
        merged = torch.where(
            importance_mask,
            wa.lerp(wb, donor_weight),
            wa.lerp(wb, donor_ratio)
        )
        
        # Clean up
        del importance, importance_mask, donor_weight, wa, wb
        
        # Return in original dtype, move to CPU
        result = merged.to(dtype=original_dtype, device="cpu")
        del merged
        
        return result
    
    def merge_checkpoints(
        self,
        base_model,
        donor_model,
        donor_ratio: float,
        donor_factor: float,
        importance_threshold: float = 0.1,
        device: str = "auto"
    ) -> Tuple:
        """Merge two models using importance-weighted merging."""
        
        # Get device settings
        device_str, gpu_batch_size, cleanup_frequency = self.get_device_settings(device)
        
        if device_str == "cuda" and not torch.cuda.is_available():
            print("GPU requested but CUDA not available, falling back to CPU")
            device_str = "cpu"
            cleanup_frequency = 5
        
        torch_device = torch.device(device_str)
        
        # Get original state dicts
        state_base_orig = base_model.model_state_dict()
        state_donor_orig = donor_model.model_state_dict()
        
        # Bake LoRA/LoKR into base if present
        has_base_patches = len(base_model.patches) > 0
        if has_base_patches:
            print(f"Baking {len(base_model.patches)} patches into base model...")
            state_base = {}
            for key in state_base_orig.keys():
                weight = state_base_orig[key].clone().to(torch_device)
                if key in base_model.patches:
                    patches = base_model.patches[key]
                    weight = comfy.lora.calculate_weight(patches, weight, key, intermediate_dtype=weight.dtype)
                state_base[key] = weight.cpu()
        else:
            state_base = state_base_orig
        
        # Bake LoRA/LoKR into donor if present
        has_donor_patches = len(donor_model.patches) > 0
        if has_donor_patches:
            print(f"Baking {len(donor_model.patches)} patches into donor model...")
            state_donor = {}
            for key in state_donor_orig.keys():
                weight = state_donor_orig[key].clone().to(torch_device)
                if key in donor_model.patches:
                    patches = donor_model.patches[key]
                    weight = comfy.lora.calculate_weight(patches, weight, key, intermediate_dtype=weight.dtype)
                state_donor[key] = weight.cpu()
        else:
            state_donor = state_donor_orig
        
        # Verify models have same keys
        keys_base = set(state_base.keys())
        keys_donor = set(state_donor.keys())
        
        if keys_base != keys_donor:
            missing_in_donor = keys_base - keys_donor
            missing_in_base = keys_donor - keys_base
            error_msg = "Models have mismatched keys:\n"
            if missing_in_donor:
                error_msg += f"Missing in donor: {list(missing_in_donor)[:5]}\n"
            if missing_in_base:
                error_msg += f"Missing in base: {list(missing_in_base)[:5]}\n"
            raise ValueError(error_msg)
        
        total_keys = len(state_base)
        
        print(f"\n{'='*60}")
        print(f"Subtle Merge: Merging {total_keys} tensors")
        print(f"  donor_ratio={donor_ratio}, donor_factor={donor_factor}")
        print(f"  importance_threshold={importance_threshold}")
        if device_str == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  Device: GPU ({vram_gb:.1f}GB VRAM)")
            print(f"  Batch size: {gpu_batch_size}, Cleanup: every {cleanup_frequency} tensors")
        else:
            print(f"  Device: CPU")
            print(f"  Cleanup: every {cleanup_frequency} tensors")
        print(f"{'='*60}\n")
        
        # Clone base model
        merged_model = base_model.clone()
        patches = {}
        
        # Progress tracking
        processed = 0
        last_percent = -1
        
        def print_progress(current, total):
            nonlocal last_percent
            percent = int((current / total) * 100)
            if percent != last_percent and percent % 5 == 0:
                bar_length = 40
                filled = int(bar_length * current / total)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\r  Progress: [{bar}] {percent}% ({current}/{total})", end='', flush=True)
                last_percent = percent
        
        # GPU: batch processing
        if device_str == "cuda":
            batch = []
            batch_keys = []
            
            try:
                for i, key in enumerate(state_base.keys(), 1):
                    weight_base = state_base[key]
                    weight_donor = state_donor[key]
                    
                    if not isinstance(weight_base, torch.Tensor):
                        continue
                    
                    if weight_base.shape != weight_donor.shape:
                        print(f"\n  Warning: Shape mismatch for {key}, using base model")
                        continue
                    
                    batch.append((weight_base, weight_donor))
                    batch_keys.append(key)
                    
                    if len(batch) >= gpu_batch_size:
                        for batch_key, (wb, wd) in zip(batch_keys, batch):
                            merged_weight = self.importance_merge(
                                wb, wd, donor_ratio, donor_factor, importance_threshold, device_str
                            )
                            patches[batch_key] = ("set", (merged_weight,))
                            processed += 1
                            print_progress(processed, total_keys)
                        
                        batch = []
                        batch_keys = []
                        torch.cuda.empty_cache()
                        if i % cleanup_frequency == 0:
                            gc.collect()
                
                # Process remaining
                if batch:
                    for batch_key, (wb, wd) in zip(batch_keys, batch):
                        merged_weight = self.importance_merge(
                            wb, wd, donor_ratio, donor_factor, importance_threshold, device_str
                        )
                        patches[batch_key] = ("set", (merged_weight,))
                        processed += 1
                        print_progress(processed, total_keys)
                    
                    torch.cuda.empty_cache()
                    gc.collect()
            
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # CPU: one at a time
        else:
            try:
                for i, key in enumerate(state_base.keys(), 1):
                    weight_base = state_base[key]
                    weight_donor = state_donor[key]
                    
                    if not isinstance(weight_base, torch.Tensor):
                        continue
                    
                    if weight_base.shape != weight_donor.shape:
                        print(f"\n  Warning: Shape mismatch for {key}, using base model")
                        continue
                    
                    merged_weight = self.importance_merge(
                        weight_base,
                        weight_donor,
                        donor_ratio,
                        donor_factor,
                        importance_threshold,
                        device_str
                    )
                    
                    patches[key] = ("set", (merged_weight,))
                    del merged_weight
                    processed += 1
                    print_progress(processed, total_keys)
                    
                    if i % cleanup_frequency == 0:
                        gc.collect()
            
            finally:
                gc.collect()
        
        # Add patches
        merged_model.add_patches(patches, strength_patch=1.0)
        
        print(f"\n\n{'='*60}")
        print("Subtle Merge: Complete!")
        print(f"{'='*60}\n")
        
        return (merged_model,)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "SubtleMerge": SubtleMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SubtleMerge": "Subtle Merge"
}
