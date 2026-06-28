# SAR 预训练代码

本目录只包含预训练、环境配置、启动脚本和少量冒烟测试图像。建议在 Linux + NVIDIA GPU 环境运行。

## 1. 硬件与软件

推荐配置：

- 4×NVIDIA A100（40GB 或 80GB 均可）
- Linux x86_64
- NVIDIA Driver 支持 CUDA 12.1
- Conda 或 Mamba
- 充足的本地 SSD 空间；单个完整 checkpoint 可能超过 1GB

创建环境：

```bash
conda env create -f environment.yml
conda activate sar-pretrain
```

如果服务器只能使用已有 PyTorch 环境，请先安装与服务器 CUDA 匹配的 PyTorch/torchvision，再执行：

```bash
pip install -r requirements.txt
```

检查 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name()); print(torch.cuda.is_bf16_supported())"
```

## 2. 数据目录

数据加载器递归读取 PNG、JPG、JPEG、TIF、TIFF 和 BMP，并统一转换为单通道。类别目录不是必需的。

```text
dataset/
├── source_a/
│   ├── image_0001.png
│   └── image_0002.png
└── source_b/
    └── image_0003.tif
```

仓库附带的 `dataset/SARSim` 仅用于冒烟测试，不能替代正式预训练数据。

## 3. 先运行冒烟测试

单 GPU：

```bash
bash scripts/smoke_pretrain.sh
```

CPU 仅用于验证代码路径：

```bash
bash scripts/smoke_pretrain_cpu.sh
```

## 4. 探测 4×A100 最大 batch size

最大 batch size 取决于 A100 显存容量、驱动、PyTorch、是否编译扩展以及同卡是否存在其他进程，不能写死。以下命令会用合成输入和完整前向/反向/优化器步骤进行二分探测：

```bash
python scripts/find_max_batch_size.py --gpus 4 --upper 1024
```

输出示例：

```text
MAX_BATCH_SIZE_PER_GPU=160
RECOMMENDED_BATCH_SIZE_PER_GPU=144
GLOBAL_BATCH_AT_MAX=640
```

`MAX` 是探测时的容量上限；长时间训练建议使用 `RECOMMENDED`，为数据加载波动和显存碎片预留约 10%。如果必须使用容量上限，可直接采用 `MAX`。

## 5. 4×A100 正式预训练

```bash
export BATCH_SIZE=128
export DATA_PATH=/path/to/full_dataset
export OUTPUT_DIR=/path/to/output
bash scripts/pretrain_4xa100.sh
```

脚本使用：

- 4 进程 DistributedDataParallel；
- BF16 自动混合精度；
- TF32 矩阵计算；
- 每 GPU 独立 batch size；
- 自动按全局 batch size 缩放学习率。

全局 batch size 为：

```text
BATCH_SIZE × 4 × accum_iter
```

从 checkpoint 恢复：

```bash
torchrun --standalone --nproc_per_node=4 Pretraining/main_pretrain.py \
  --data_path /path/to/full_dataset \
  --output_dir /path/to/output \
  --log_dir /path/to/output \
  --batch_size 128 \
  --amp_dtype bf16 \
  --resume /path/to/checkpoint.pth
```

## 6. 可选编译相对位置编码扩展

不编译也能运行；编译后通常更快。必须在目标服务器和最终 PyTorch 环境中执行：

```bash
cd Pretraining/rpe_ops
python setup.py build_ext --inplace
cd ../..
```

编译失败时可继续使用 Python 回退实现。

## 7. 常见问题

### CUDA out of memory

重新探测或降低每 GPU batch：

```bash
export BATCH_SIZE=64
bash scripts/pretrain_4xa100.sh
```

### 数据集小于全局 batch

训练使用 `drop_last=True`。正式数据的图像数量必须不小于全局 batch，否则一个 epoch 可能没有有效 batch。

### NCCL 初始化失败

确认四张 GPU 可见：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nvidia-smi
```

单机脚本使用 `torchrun --standalone`，不需要手工设置主节点地址。

### TensorBoard

```bash
tensorboard --logdir /path/to/output --port 6006
```
