# ComfyUI-SubtleNodes
Custom Nodes for ComfyUI

<br>

*PowerUp (DARE) Merger*

Offers the DARE merge feature of the A1111 UntitledMerger extension. It adds the difference of the donor_model to the base_model. *Drop_rate* defines how much will be added. The higher the value, the more. *Addition_multiplier* adjusts the strength of the addition. Lower values don't alter the base_model as much & integrate new capabilities better. *Seed* randomizes the added differences. Best set it to *fixed*. The six merge switches let you deactivate the addition to input, mid & output blocks as well as attention, convolution & normalization layers.

<br>

**Installation**

Copy the file into the custom_nodes sub folder of Comfy. Don't create a sub folder for it there. Restart ComfyUI.
