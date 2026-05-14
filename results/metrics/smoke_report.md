# First Benchmark Report

## Train/Validation Split

- Seed: `42`
- Validation fraction from official training sequences: 10%
- Train sequences (19): GOPR0372_07_00, GOPR0372_07_01, GOPR0374_11_01, GOPR0374_11_02, GOPR0378_13_00, GOPR0379_11_00, GOPR0380_11_00, GOPR0384_11_01, GOPR0384_11_02, GOPR0384_11_03, GOPR0384_11_04, GOPR0385_11_00, GOPR0386_11_00, GOPR0477_11_00, GOPR0857_11_00, GOPR0868_11_01, GOPR0868_11_02, GOPR0871_11_01, GOPR0881_11_00
- Validation sequences (3): GOPR0374_11_00, GOPR0374_11_03, GOPR0884_11_00

## Model

- Architecture: `UNetDeblurModel`
- Trainable parameters: 209,827
- Total parameters: 209,827
- Base channels: 8
- Device used: `cpu`

## Training Setup

- Patch size: 256x256
- Batch size: 2
- Epochs: 1
- Optimizer: Adam
- Learning rate: 0.0002
- Scheduler: StepLR(step_size=4, gamma=0.5)
- Loss: L1

## Metrics

- Identity validation: PSNR 24.0336, SSIM 0.7350
- Identity test: PSNR 25.6401, SSIM 0.7930
- U-Net validation: PSNR 24.0389, SSIM 0.7352
- U-Net test: PSNR 25.6493, SSIM 0.7933
- Validation improvement over identity: PSNR +0.0053, SSIM +0.0002
- Test improvement over identity: PSNR +0.0092, SSIM +0.0003

## Training History

- Epoch 1: train_loss 0.0424, train_psnr 25.5428, val_loss 0.0395, val_psnr 24.0389, lr 0.000200, time 591.8s

## Qualitative Observations

- The identity baseline preserves blur, so it gives a realistic lower bound for restoration quality.
- The first U-Net benchmark mainly sharpens strong motion streaks and edges; remaining failure cases should guide the next round of model improvements.

## Issues Before Stronger Models

- This run uses a modest first-pass U-Net, so model capacity and training duration are still conservative.
- Validation is derived from only three sequences, so conclusions should be treated as directional rather than final.
- Full-resolution evaluation is deterministic, but training still samples one random patch per frame per epoch; longer runs should improve stability.
