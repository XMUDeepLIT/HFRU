# Object Hallucination-Free Reinforcement Unlearning for Vision-Language Models

## Table of Contents

- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Testing](#testing)
- [Visualization](#visualization)

## Requirements

Use separate Conda environments for each stage to avoid dependency conflicts.

### 1) Stage-1 Training (SFT)

```bash
conda create -n llamafactory python=3.12
conda activate llamafactory
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install -r requirements-llamafactory.txt
```

### 2) Stage-2 Training (RL) and Testing

```bash
conda create -n verl python=3.10
conda activate verl
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r requirements-verl.txt
```

### 3) Visualization

```bash
conda create -n visualization python=3.10
conda activate visualization
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r requirements-visualization.txt
```

## Data Preprocessing

Prepare the training data before starting model training.

### PACS

```bash
bash ./scripts/pacs/pacs_process.sh
```

### VGGFace2

```bash
bash ./scripts/vgg/vgg_process.sh
```

## Training

Run the following scripts for two-stage training.

### PACS

```bash
bash ./scripts/pacs/pacs_train_stage1.sh
bash ./scripts/pacs/pacs_train_stage2.sh
```

### VGGFace2

```bash
bash ./scripts/vgg/vgg_train_stage1.sh
bash ./scripts/vgg/vgg_train_stage2.sh
```

> [!IMPORTANT]
> If the reward stops improving during stage-2 RL training, run the script below to replace non-weight files and then resume training:
>
> ```bash
> bash ./scripts/replace_non_weight_files.sh
> ```

## Testing

Run the following scripts to evaluate model performance.

### PACS

```bash
bash ./scripts/pacs/pacs_test.sh
```

### VGGFace2

```bash
bash ./scripts/vgg/vgg_test.sh
```

## Visualization

If you only need visualization for the face recognition scenario, you can directly use the provided checkpoint at `./model/qwen_to_clip_projector.pt`:

```bash
bash ./scripts/visualization/test.sh
```

If you want visualization support for other scenarios, train the mapping network first:

```bash
bash ./scripts/visualization/train.sh
```

## 🙏 Acknowledgements

Our framework builds upon the excellent work of:
- [**LlamaFactory**](https://github.com/hiyouga/LLaMAFactory)
- [**verl**](https://github.com/verl-project/verl)
- [**IP-Adapter**](https://github.com/tencent-ailab/IP-Adapter)