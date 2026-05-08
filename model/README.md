# Model Download Guide

This document lists required models and the expected local paths for training, testing, and visualization.

## Required Models

### Training
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

### Testing
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

### Visualization
- [IP-Adapter](https://huggingface.co/h94/IP-Adapter)
- [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

Move the model folder from IP-Adapter to the runtime path expected by the project:

```bash
mv ./model/IP-Adapter/models ./src/IP-Adapter/models
mv ./model/IP-Adapter/sdxl_models ./src/IP-Adapter/sdxl_models
```