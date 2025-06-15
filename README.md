# Vision Transformers on Cifar10

Thank you to Kentaro Yoshioka for the original repository playground that this fork is based on.

```
@misc{yoshioka2024visiontransformers,
  author       = {Kentaro Yoshioka},
  title        = {vision-transformers-cifar10: Training Vision Transformers (ViT) and related models on CIFAR-10},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/kentaroy47/vision-transformers-cifar10}}
}
```

# My work

### (1) Fixed compatibility with multiprocessing on macOS

On macOS, Python’s multiprocessing module requires that the main script logic is placed under if **name** == "**main**" to avoid recursive process spawning. This fix ensures the script runs correctly when using DataLoader with num_workers > 0.

### (2) Added support for torch.mps on Apple Silicon (M1/M2)

See results of using MPS instead of CPU in section `(A) Training performance CPU vs MPS`

### (3) Added Support for [Adan Optimizer](https://github.com/lucidrains/Adan-pytorch)

Integrated Adan via `adan-pytorch` and compared two learning rates on `vit_small`. See baseline results for the integration in section `(B) Training performance comparison using Adan`.

### (4) Hyperparameter Sweep for Adan Optimizer

To find good hyperparameters for the Adan optimizer on ViT with CIFAR-10, I ran a WandB sweep using Bayesian optimization (see full configuration at `sweeps/adan_betas_decay.yaml`)

The sweep tested different values for:

- learning rate (lr)
- the three beta values (adan_beta1, adan_beta2, adan_beta3)
- weight decay (weight_decay)

The value ranges were based on the original Adan paper, and quantized distributions were used to avoid overly small or noisy values. Each run trained for 5 epochs. I also used early stopping (Hyperband) to skip bad runs faster. The goal was to maximize validation accuracy.

See results for sweep in section `(C) Top 3 Sweep Results`.

### (5) Added support for adamw

### (6) 200-Epoch Training Comparison

Benchmarked Adan vs Adam vs AdamW using best sweep parameters.
See results at https://api.wandb.ai/links/moorekevin-/q9h9sjbq

# Results

For all results, I trained on Apple Silicon M2 Pro, and used the `vit_small` model

### (A) Training performance CPU vs MPS

Specifications:

- Epochs: 1
- lr: 1e-4

| Metric                                | CPU                    | MPS                   |
| ------------------------------------- | ---------------------- | --------------------- |
| **Epoch Duration**                    | ~10 minutes 40 seconds | ~1 minute 28 seconds  |
| **Step Time (Initial)**               | ~6.6s per step         | ~0.9s per step        |
| **Step Time (Stabilized)**            | ~6.5s per step         | ~0.91s per step       |
| **Final Training Accuracy (Epoch 0)** | 15.05%                 | 16.04%                |
| **Validation Step Time**              | ~387ms                 | ~58–60ms              |
| **Final Validation Accuracy**         | 34.82%                 | 34.55%                |
| **Speedup (Epoch Time)**              |                        | ~7.2x faster          |
| **Backend**                           | `torch.device("cpu")`  | `torch.device("mps")` |

### (B) Training performance comparison using Adan

Specifications:

- Ran on `torch.mps`
- Epochs: 1
- Optimizer: Adan

| Metric                        | Adan (lr = 1e-4)     | Adan (lr = 1e-3)     |
| ----------------------------- | -------------------- | -------------------- |
| **Epoch Duration**            | ~1 minute 29 seconds | ~1 minute 30 seconds |
| **Final Training Accuracy**   | 13.84%               | 18.13%               |
| **Validation Step Time**      | ~58ms                | ~57ms                |
| **Final Validation Accuracy** | 28.52%               | 38.27%               |
| **Validation Loss**           | 198.64               | 171.08               |
| **Optimizer Source**          | `adan-pytorch`       | `adan-pytorch`       |

> **Note**: Increasing the learning rate from 1e-4 to 1e-3 significantly improved convergence after a single epoch.

### (C) Top 3 Sweep Results

Specifications:

- Ran on `torch.mps`
- Epochs: 5
- Optimizer: Adan

> **Note**: See sweep file under `sweeps/adan_betas_decay.yaml`

| Rank  | val_acc | lr     | adan_beta1 | adan_beta2 | adan_beta3 | weight_decay |
| ----- | ------- | ------ | ---------- | ---------- | ---------- | ------------ |
| Top 1 | 53.01%  | 0.0009 | 0.035      | 0.11       | 0.015      | 0.02         |
| Top 2 | 52.64%  | 0.0011 | 0.035      | 0.09       | 0.015      | 0.02         |
| Top 3 | 52.62%  | 0.0009 | 0.04       | 0.12       | 0.02       | 0.015        |

## Original results from base directory (not my results)

|                                                                           | Accuracy |                                                                                 Train Log                                                                                  |
| :-----------------------------------------------------------------------: | :------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                ViT patch=2                                |   80%    |                                                                                                                                                                            |
|                           ViT patch=4 Epoch@200                           |   80%    | [Log](https://wandb.ai/arutema47/cifar10-challange/reports/Untitled-Report--VmlldzoxNjU3MTU2?accessToken=3y3ib62e8b9ed2m2zb22dze8955fwuhljl5l4po1d5a3u9b7yzek1tz7a0d4i57r) |
|                           ViT patch=4 Epoch@500                           |   88%    | [Log](https://wandb.ai/arutema47/cifar10-challange/reports/Untitled-Report--VmlldzoxNjU3MTU2?accessToken=3y3ib62e8b9ed2m2zb22dze8955fwuhljl5l4po1d5a3u9b7yzek1tz7a0d4i57r) |
|                                ViT patch=8                                |   30%    |                                                                                                                                                                            |
|                                 ViT small                                 |   80%    |                                                                                                                                                                            |
|                                 MLP mixer                                 |   88%    |                                                                                                                                                                            |
|                                   CaiT                                    |   80%    |                                                                                                                                                                            |
|                                  Swin-t                                   |   90%    |                                                                                                                                                                            |
|                         ViT small (timm transfer)                         |  97.5%   |                                                                                                                                                                            |
|                         ViT base (timm transfer)                          |  98.5%   |                                                                                                                                                                            |
| [ConvMixerTiny(no pretrain)](https://openreview.net/forum?id=TVHS5Y4dNvM) |  96.3%   |    [Log](https://wandb.ai/arutema47/cifar10-challange/reports/convmixer--VmlldzoyMjEyOTk1?accessToken=2w9nox10so11ixf7t0imdhxq1rf1ftgzyax4r9h896iekm2byfifz3b7hkv3klrt)    |
|                                 resnet18                                  |   93%    |                                                                                                                                                                            |
|                             resnet18+randaug                              |   95%    | [Log](https://wandb.ai/arutema47/cifar10-challange/reports/Untitled-Report--VmlldzoxNjU3MTYz?accessToken=968duvoqt6xq7ep75ob0yppkzbxd0q03gxy2apytryv04a84xvj8ysdfvdaakij2) |

# Instructions from original repository

## Usage example

`python train_cifar10.py` # vit-patchsize-4

`python train_cifar10.py  --size 48` # vit-patchsize-4-imsize-48

`python train_cifar10.py --patch 2` # vit-patchsize-2

`python train_cifar10.py --net vit_small --n_epochs 400` # vit-small

`python train_cifar10.py --net vit_timm` # train with pretrained vit

`python train_cifar10.py --net convmixer --n_epochs 400` # train with convmixer

`python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3`

`python train_cifar10.py --net cait --n_epochs 200` # train with cait

`python train_cifar10.py --net swin --n_epochs 400` # train with SwinTransformers

`python train_cifar10.py --net res18` # resnet18+randaug

## Model Export

This repository supports exporting trained models to ONNX and TorchScript formats for deployment purposes. You can export your trained models using the `export_models.py` script.

### Basic Usage

```bash
python export_models.py --checkpoint path/to/checkpoint --model_type vit --output_dir exported_models
```
