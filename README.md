# Fossil Restoration Through Generative Deep Learning

**MSc Artificial Intelligence | University of Essex | 2025**

This repository contains the implementation of a deep learning pipeline for reconstructing damaged fossil specimens using Stable Diffusion with LoRA fine-tuning. The project explores whether pre-trained diffusion models can be adapted for paleontological reconstruction with limited training data (465 specimens).

The model achieves convergence within 2 training epochs, with a PSNR of 32.95 dB and SSIM of 0.9444. By working with 2D views instead of direct 3D generation, this approach circumvents the resolution limitations of voxel-based methods, maintaining morphological detail that would otherwise be lost. The pipeline uses depth estimation to generate partial 3D representations from reconstructed 2D views.

## Overview

This project demonstrates a complete pipeline for automated fossil reconstruction using diffusion-based deep learning and depth estimation.

### Components

| File                         | Description                                                          | Link                                                                                         |
| ---------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `fossil-inpainting-sd.ipynb` | Kaggle notebook for fine-tuning Stable Diffusion on ammonite fossils | -                                                                                            |
| `to_3d/estimate.py`          | Depth estimation and 3D generation using Depth Anything V2           | -                                                                                            |
| `dataset.zip`                | 465 ammonite specimens                                               | [Access](https://drive.google.com/file/d/1ADHKe8hkKHccMcyGnCBiyZCLnl1mEB3A/view?usp=sharing) |
| `sd_fossil_lora.zip`         | Fine-tuned LoRA weights for Stable Diffusion                         | [Access](https://drive.google.com/file/d/1Te3pJn9gjz5sdj8qyhXVL5FbDphEUaXU/view?usp=sharing) |

## Usage

Extract the dataset and LoRA weights from their respective zip files. The training notebook can be run on Kaggle.

**3D Reconstruction**

To generate 3D models from reconstructed fossil images:

Install dependencies:

```bash
cd to_3d
pip install -r requirements.txt
```

Edit `estimate.py` and replace `IMAGE_PATH` with your inpainted fossil image path

Run the reconstruction:

```bash
python estimate.py
```

This will generate a depth map and corresponding 3D mesh from your reconstructed fossil image.
