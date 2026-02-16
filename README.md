# ComfyUI-SubtleNodes
Custom Nodes for ComfyUI

<br><img width="1283" height="713" alt="subtlenodes" src="https://github.com/user-attachments/assets/40ffcab4-28ac-499c-bb38-930e1ce79adc" />


*DARE Merge*

DARE merge randomly ignores parts of the weight difference between base_model and donor_model according to a dropout rate & seed, then scales and adds the remaining differences to the base_model. Simply adding all differences would turn the base_model into the donor_model, which would be of no benefit.

*Drop_rate* randomly discards a percentage of weight differences between the models. Higher values keep less differences but amplifiy them more thus making the resulting model more distinct from the base_model.
*Addition_multiplier* controls the strength of the added changes. Lower values blend more subtly, higher values increase the donor’s influence, and negative values subtract the donor’s traits.
*Seed* sets the randomization pattern for dropped weight differences. Keep it fixed for reproducible results.
The six merge sliders control how merging is applied to input, mid and output blocks as well as attention, convolution and normalization layers. A value of zero skips the block or layer, values between 0.01 and 0.99 reduce the added difference, 1.0 does default processing and values of 1.01 to 2.0 amplify the added difference.

Increase *drop_rate* to add as much of donor_model as possible until side effects show up. Then increase *addition_multiplier* for more of donor or decrease to restore the old look of the base_model. Try different seeds if you are not satisfied. The goal is to add enough new traits while avoiding problems.

Double click on the background in ComfyUI and enter powerup or dare to find it. Add two Load Checkpoint nodes and connect them to the inputs of this node. Connect the node's output to your usual workflow as you would do with a Load Checkpoint node. Connect the output to a Save Checkpoint node to save the new checkpoint.

This node was only tested with SDXL checkpoints so far.

<br>

*Subtle Merge*

Subtle Merge intelligently combines two checkpoints by analyzing how much each weight changed between the base and donor models. Weights that changed significantly are considered "important" and get merged more strongly toward the donor, while minor changes are treated as noise and blended normally. This preserves the donor's key innovations without averaging everything equally.

*Donor_ratio* controls the baseline blend strength - how much donor to use overall. Zero means pure base model, one means pure donor model, and 0.5 is a fifty-fifty mix. 
*Donor_factor* controls how much extra influence important weights get. At 1.0, highly important weights can reach up to 100% donor influence regardless of the donor_ratio setting. At 5.0 a lot of weights are transferred from the donor model at full strength. At 0.0, all weights are treated equally like a simple average merge. 
*Importance_threshold* filters out noise. Only weights that changed more than this amount get the importance boost, while everything else gets basic blending.

Start with *donor_ratio* at 0.5 and *donor_factor* at 1.0 for a balanced merge that respects important changes. If the result is too subtle, increase *donor_factor* to pull more from the donor overall. If the merge feels too aggressive, lower *donor_factor* to reduce how much important weights are amplified. The node automatically detects your GPU's VRAM and optimizes processing speed.

**Usage**

Double click on the background in ComfyUI and enter DARE or Subtle to find each node. Add two Load Checkpoint or Load Diffusion Model nodes and connect them to the inputs. Connect the node's output to your usual workflow as you would do with a Load Checkpoint node. Connect the output to a Save Checkpoint/Model node to save the new checkpoint.
Both nodes automatically bake LoRA/LoKR patches before merging, so LoRAs/LoKRs get merged too.

**Installation**

Copy both file into the *custom_nodes* sub folder of ComfyUI. Don't create a sub folder for them there. Restart ComfyUI.
