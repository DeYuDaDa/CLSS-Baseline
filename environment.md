# 复现四个实验所需的Python环境配置

## 实验概述

本次需要复现的四个实验分别是：
1. Spherical-Steering
2. SEAL (Steerable Reasoning Calibration)
3. CCA (Contrastive Activation Addition)
4. Manifold Steering

## 统一Python环境配置

### 基础环境
- **Python版本**: 3.10+
- **操作系统**: 支持Windows、Linux、macOS
- **硬件要求**: 建议使用GPU加速，至少16GB内存

### 依赖包安装

#### 方法一：使用pip安装

```bash
# 基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.57.1
pip install numpy==2.2.6 scipy==1.15.2 scikit-learn==1.7.2

# SEAL特定依赖
pip install antlr4-python3-runtime==4.11.1 sympy==1.12 mpmath==1.3.0

# Spherical-Steering特定依赖
pip install vllm==0.11.2 xformers==0.0.33.post1
pip install datasets==4.4.1 evaluate==0.4.6
pip install accelerate==1.12.0

# 通用工具库
pip install tqdm==4.67.1 pyyaml==6.0.3
pip install matplotlib==3.10.7
pip install huggingface-hub==0.36.0
```

#### 方法二：使用conda环境

```bash
# 创建环境
conda create -n steering-experiments python=3.10
conda activate steering-experiments

# 安装PyTorch
conda install torch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装其他依赖
conda install -c conda-forge numpy scipy scikit-learn
pip install transformers==4.57.1
pip install antlr4-python3-runtime==4.11.1 sympy==1.12 mpmath==1.3.0
pip install vllm==0.11.2 xformers==0.0.33.post1
pip install datasets==4.4.1 evaluate==0.4.6
pip install accelerate==1.12.0
pip install tqdm==4.67.1 pyyaml==6.0.3
pip install matplotlib==3.10.7
pip install huggingface-hub==0.36.0
```

## 各实验特定配置

### 1. Spherical-Steering
- **主要文件**: `src/Spherical-Steering-main/Spherical-Steering-main/`
- **运行方式**: 
  - 向量生成: `bash quickstart_qwen.sh` 或 `bash quickstart_llama.sh`
  - 评估: `python eval_math_spherical.py`

### 2. SEAL
- **主要文件**: `src/SEAL-main/SEAL-main/`
- **运行方式**:
  - 提取引导向量: `bash scripts/generate_vector.sh`
  - 运行引导: `bash scripts/steering.sh`
  - 评估: `python eval_MATH_steering.py` 或 `python eval_code_steering.py`

### 3. CCA
- **主要文件**: `src/CCA/`
- **运行方式**:
  - 评估: `python eval_caa_math.py`

### 4. Manifold Steering
- **主要文件**: `src/manifold/`
- **运行方式**:
  - 提取激活值: `python extract_all_layers.py`
  - 评估: `python eval_math_manifold.py`

## 注意事项

1. **模型下载**: 实验需要使用预训练模型，如Llama 2、Qwen等，请确保有足够的磁盘空间和网络带宽。

2. **GPU内存**: 运行这些实验需要较大的GPU内存，建议使用至少16GB显存的GPU。

3. **依赖版本**: 某些依赖包的版本可能需要根据实际情况进行调整，特别是PyTorch和CUDA版本需要匹配。

4. **环境变量**: 建议设置以下环境变量以优化性能：
   - `CUDA_VISIBLE_DEVICES`: 指定使用的GPU
   - `PYTORCH_CUDA_ALLOC_CONF`: 优化CUDA内存分配

5. **数据准备**: 实验需要一些数据集，如MATH、GSM等，请确保数据集已正确下载并放置在指定位置。

## 故障排除

- **CUDA内存不足**: 可以尝试使用`device_map="auto"`或降低batch size。
- **依赖冲突**: 如有依赖冲突，建议使用虚拟环境或容器化解决方案。
- **模型加载失败**: 确保Hugging Face的token已正确配置，且网络连接正常。

## 总结

以上配置涵盖了复现四个实验所需的Python环境。建议在运行实验前，先检查所有依赖是否已正确安装，并确保硬件资源满足要求。