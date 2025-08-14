# ComfyUI-SubtleNodes
Custom Nodes for ComfyUI

<br>

*PowerUp (DARE) Merge*

Offers the DARE merge feature of A1111's [UntitledMerger](https://github.com/groinge/sd-webui-untitledmerger) extension. It randomly ignores parts of the weight difference between base_model and donor_model according to a dropout rate & seed, then scales and adds the remaining differences to the base_model. Simply adding all differences would turn the base_model into the donor_model, which would be of no benefit.

*Drop_rate* randomly discards a percentage of weight differences between the models. Higher values use more randomness, making the resulting model more distinct from the base_model.
*Addition_multiplier* controls the strength of the added changes. Lower values blend more subtly, higher values amplify the donor’s influence, and negative values subtract the donor’s traits.
*Seed* sets the randomization pattern for dropped weights. Keep it fixed for reproducible results.
The six merge switches control whether merging is applied to input, mid, and output blocks, and whether it affects attention, convolution and normalization layers.

Add two Load Checkpoint nodes and connect them to the inputs of this node. Connect the node's output to your usual workflow as you would do with a Load Checkpoint node. Connect the output to a Save Checkpoint node to save the new checkpoint.
This node Was only tested with SDXL so far.

<br>

**Installation**

Copy the file into the *custom_nodes* sub folder of ComfyUI. Don't create a sub folder for it there. Restart ComfyUI.
