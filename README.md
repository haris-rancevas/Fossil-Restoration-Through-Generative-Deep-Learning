# Fossil Restoration Through Generative Deep Learning

**MSc Artificial Intelligence Dissertation** | University of Essex | 2025

This repository contains the implementation, dataset, and fine-tuned model for the project "Fossil Restoration Through Generative Deep Learning", which applies Stable Diffusion with LoRA fine-tuning to reconstruct damaged fossil images and uses depth estimation to generate corresponding 3D structures.

## Overview

This project demonstrates a complete pipeline for automated fossil reconstruction using diffusion-based deep learning and depth estimation.

It includes the following key components:

- **`fossil-inpainting-sd.ipynb`** — Kaggle notebook used to **fine-tune a Stable Diffusion Inpainting model** for ammonite fossil reconstruction.

- **`estimate.py`** — Python script that uses **Depth Anything V2** to estimate depth and **reconstruct 3D fossil structures** from the 2D inpainted images.

- **`dataset`** — Contains fossil specimen data sourced from the **GB3D Type Fossil Repository**, focusing on **ammonite fossils** from the order _Ammonitida_. Access [here](https://drive.google.com/file/d/1ADHKe8hkKHccMcyGnCBiyZCLnl1mEB3A/view?usp=sharing).

- **`sd_fossil_lora`** — Directory containing the **fine-tuned LoRA weights** and configurations for Stable Diffusion.

## Project Structure

```bash
├── fossil-inpainting-sd.ipynb # Kaggle notebook for Stable Diffusion LoRA fine-tuning
└── to_3d/
│    ├── estimate.py # 3D reconstruction using Depth Anything
│    └── requirements.txt
```

### Fine-tuned LoRA weights

Access directory containing the **fine-tuned LoRA weights** and configurations for Stable Diffusion [here](https://drive.google.com/file/d/1Te3pJn9gjz5sdj8qyhXVL5FbDphEUaXU/view?usp=sharing).

```bash
└── sd_fossil_lora/
    ├── epoch_10/
    │    ├── model.safetensors # Final fine-tuned LoRA weights
    │    ├── optimizer.bin
    │    ├── scaler.pt
    │    └──  random_states_0.pkl
    └── fossil_lora_unet/
        ├── adapter_model.safetensors # LoRA adapter weights
        └── adapter_config.json # LoRA configuration
```
