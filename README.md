# Fossil Restoration Through Generative Deep Learning

**MSc Artificial Intelligence | University of Essex | 2025**

This repository contains the implementation of a deep learning pipeline for reconstructing damaged fossil specimens using Stable Diffusion with LoRA fine-tuning. The project explores whether pre-trained diffusion models can be adapted for paleontological reconstruction with limited training data (465 specimens).

The model achieves convergence within 2 training epochs, with a PSNR of 32.95 dB and SSIM of 0.9444. By working with 2D views instead of direct 3D generation, this approach circumvents the resolution limitations of voxel-based methods, maintaining morphological detail that would otherwise be lost. The pipeline uses depth estimation to generate partial 3D representations from reconstructed 2D views.

## Overview

The repository provides end-to-end code for 2D inpainting and 3D depth reconstruction.

### Components

| File                         | Description                                                          | Link                                                                                         |
| ---------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `fossil-inpainting-sd.ipynb` | Kaggle notebook for fine-tuning Stable Diffusion on ammonite fossils | -                                                                                            |
| `to_3d/estimate.py`          | Depth estimation and 3D generation using Depth Anything V2           | -                                                                                            |
| `dataset.zip`                | 465 ammonite specimens                                               | [Access](https://drive.google.com/file/d/1ADHKe8hkKHccMcyGnCBiyZCLnl1mEB3A/view?usp=sharing) |
| `sd_fossil_lora.zip`         | Fine-tuned LoRA weights for Stable Diffusion                         | [Access](https://drive.google.com/file/d/1Te3pJn9gjz5sdj8qyhXVL5FbDphEUaXU/view?usp=sharing) |

## Usage

The training notebook (`fossil-inpainting-sd.ipynb`) shows the fine-tuning process used on Kaggle with the provided dataset. The LoRA weights are provided but require separate inference setup.

**3D Reconstruction**

To generate 3D models from reconstructed fossil images:

Create [conda](https://anaconda.org/anaconda/conda) environment (isolates dependencies):

```bash
conda create -n myenv python=3.10.15
conda activate myenv
```

Install dependencies:

```bash
cd to_3d
pip install -r requirements.txt
```

Run the reconstruction:

```bash
python estimate.py --image path/to/image.png --detail 1 --elevation 90 --azimuth 90
```

Parameters:

- `--image`: Path to input image (default: reconstructed.png)
- `--detail`: Visualization detail level (1=max, 2=half, 3=third, etc.) (default: 1)
- `--elevation`: Default elevation angle for 3D view (default: 92)
- `--azimuth`: Default azimuth angle for 3D view (default: 90)

This will generate a depth map and corresponding 3D mesh from your reconstructed fossil image.

**Sample Outputs**

![Figure 1: 3D Surface Reconstruction](to_3d/sample_outputs/reconstruction.png)
_Figure 1: 3D surface reconstruction from depth estimation_.

![Figure 2: Depth Map](to_3d/sample_outputs/depth_map.png)
_Figure 2: Estimated depth map showing surface topology_.

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{rancevas2025generativeinpaintingfossils,
  title={Fossil Restoration Through Generative Deep Learning},
  author={Rancevas, H.},
  year={2025},
  school={University of Essex},
  type={MSc Dissertation},
  note={MSc Artificial Intelligence}
}
```
