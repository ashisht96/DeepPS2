# DeepPS2
[DeepPS2: Revisiting Photometric Stereo using Two Differently Illuminated Images](https://arxiv.org/abs/2207.02025)

[Ashish Tiwari](https://sites.google.com/iitgn.ac.in/ashishtiwari/home) and [Shanmuganathan Raman](https://iitgn.ac.in/faculty/cse/shanmuganathan)

Accepted in [ECCV 2022](https://eccv2022.ecva.net/)

This work address the PS2 problem (photometric Stereo with two images) under *uncalibrated* and *self-supervised* setting.

![alt text](https://github.com/ashisht96/DeepPS2/blob/main/images/bd.png)

# Overview

DeepPS2 is implemented in [PyTorch](https://pytorch.org/) and tested with Ubuntu (16.04).

Use the following command to run the code:

```python
python train.py --obj [object name] --gpu_ids 0 --checkpoints_dir ./checkpoints --save_dir ./visual_results
```


