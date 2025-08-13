# ComfyUI-SubtleNodes
Custom Nodes for ComfyUI

<br>

*PowerUp (DARE) Merge*

Offers the DARE merge feature of A1111's [UntitledMerger](https://github.com/groinge/sd-webui-untitledmerger) extension. It adds the difference of the donor_model (e.g. new capabilities, concepts & looks) to the base_model. *Drop_rate* defines how much will be added. The higher the value, the more. *Addition_multiplier* adjusts the strength of the addition. Lower values don't alter the base_model as much & integrate new capabilities more smoothly without changing its style. *Seed* randomizes the added differences. Best set it to *fixed*. The six merge switches let you deactivate the addition for input, mid & output blocks as well as attention, convolution & normalization layers.

<br>

**Installation**

Copy the file into the *custom_nodes* sub folder of ComfyUI. Don't create a sub folder for it there. Restart ComfyUI.
