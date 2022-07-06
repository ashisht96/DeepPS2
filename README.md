# DeepPS2
[DeepPS2: Revisiting Photometric Stereo using Two Differently Illuminated Images](https://arxiv.org/abs/2207.02025)

[Ashish Tiwari](https://sites.google.com/iitgn.ac.in/ashishtiwari/home) and [Shanmuganathan Raman](https://iitgn.ac.in/faculty/cse/shanmuganathan)

Accepted in [ECCV 2022](https://eccv2022.ecva.net/)

This work address the PS2 problem (photometric Stereo with two images) under *uncalibrated* and *self-supervised* setting.

![alt text](https://github.com/ashisht96/DeepPS2/blob/main/images/bd.png)

# Overview

DeepPS2 is implemented in [PyTorch](https://pytorch.org/) and tested with Ubuntu (16.04). 

Download the [DiLiGenT](https://sites.google.com/site/photometricstereodata/single) Dataset and extract it into the data folder.

Use the following command to run the code:

```python
python train.py --obj [object_name] --gpu_ids [gpu_id] --checkpoints_dir [path_to_save_chkpts] --save_dir [path_to_save_visual_results]
```

# Citation
If you find this code useful in your research, please consider cite:

@misc{https://doi.org/10.48550/arxiv.2207.02025,
  doi = {10.48550/ARXIV.2207.02025},
  
  url = {https://arxiv.org/abs/2207.02025},
  
  author = {Tiwari, Ashish and Raman, Shanmuganathan},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {DeepPS2: Revisiting Photometric Stereo Using Two Differently Illuminated Images},
