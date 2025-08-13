# ComfyUI-SubtleNodes
Custom Nodes for ComfyUI

<br>

*PowerUp (DARE) Merge*

Offers the DARE merge feature of A1111's [UntitledMerger](https://github.com/groinge/sd-webui-untitledmerger) extension. It randomly masks parts of the difference between base_model and donor_model (e.g. new capabilities, concepts & looks) according to a dropout rate & seed, then scales and adds the selected difference to the base_model.

*Drop_rate* defines if smaller or larger differences will be added. The higher the value, the more distinct the end result.
*Addition_multiplier* adjusts the strength of the addition. Lower values down to zero don't alter the base_model as much & integrate new capabilities more smoothly without changing the style of the base_model. Negative values remove the difference between both.
*Seed* randomizes the selected differences. Best set it to *fixed*. The six merge switches let you deactivate merging of input, mid & output blocks as well as attention, convolution & normalization layers.

<br>

**Installation**

Copy the file into the *custom_nodes* sub folder of ComfyUI. Don't create a sub folder for it there. Restart ComfyUI.
