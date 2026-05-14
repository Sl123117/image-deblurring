# First Benchmark Report

## Run

- Artifact prefix: `l4-base48-20260424-01`

## Train/Validation Split

- Seed: `42`
- Validation fraction from official training sequences: 10%
- Train sequences (19): GOPR0372_07_00, GOPR0372_07_01, GOPR0374_11_01, GOPR0374_11_02, GOPR0378_13_00, GOPR0379_11_00, GOPR0380_11_00, GOPR0384_11_01, GOPR0384_11_02, GOPR0384_11_03, GOPR0384_11_04, GOPR0385_11_00, GOPR0386_11_00, GOPR0477_11_00, GOPR0857_11_00, GOPR0868_11_01, GOPR0868_11_02, GOPR0871_11_01, GOPR0881_11_00
- Validation sequences (3): GOPR0374_11_00, GOPR0374_11_03, GOPR0884_11_00

## Model

- Architecture: `UNetDeblurModel`
- Trainable parameters: 7,531,587
- Total parameters: 7,531,587
- Base channels: 48
- Device used: `cuda`
- AMP enabled on CUDA: True

## Training Setup

- Patch size: 256x256
- Batch size: 16
- Eval batch size: 4
- DataLoader workers: 4
- Pin memory: True
- Persistent workers: True
- Prefetch factor: 2
- CPU threads: 4
- Epochs: 24
- Validate each epoch: True
- Checkpoint monitor: val_psnr
- Optimizer: Adam
- Learning rate: 0.0002
- Scheduler: StepLR(step_size=15, gamma=0.5)
- Loss: L1

## Metrics

- Identity validation: PSNR 24.0336, SSIM 0.7350
- Identity test: PSNR 25.6401, SSIM 0.7930
- U-Net validation: PSNR 24.3528, SSIM 0.7431
- U-Net test: PSNR 26.2952, SSIM 0.8077
- Validation improvement over identity: PSNR +0.3192, SSIM +0.0081
- Test improvement over identity: PSNR +0.6551, SSIM +0.0147

## Training History

- Epoch 1: train_loss 0.0312, train_psnr 27.2119, val_loss 0.0395, val_psnr 24.0348, ips 66.57, lr 0.000200, time 48.2s
- Epoch 2: train_loss 0.0300, train_psnr 27.5853, val_loss 0.0395, val_psnr 24.0416, ips 67.56, lr 0.000200, time 47.4s
- Epoch 3: train_loss 0.0305, train_psnr 27.4274, val_loss 0.0393, val_psnr 24.0782, ips 68.57, lr 0.000200, time 47.3s
- Epoch 4: train_loss 0.0304, train_psnr 27.5147, val_loss 0.0392, val_psnr 24.0905, ips 68.04, lr 0.000200, time 47.6s
- Epoch 5: train_loss 0.0295, train_psnr 27.6926, val_loss 0.0390, val_psnr 24.1127, ips 68.43, lr 0.000200, time 47.7s
- Epoch 6: train_loss 0.0302, train_psnr 27.5406, val_loss 0.0391, val_psnr 24.0934, ips 67.73, lr 0.000200, time 47.7s
- Epoch 7: train_loss 0.0304, train_psnr 27.5417, val_loss 0.0391, val_psnr 24.1079, ips 67.62, lr 0.000200, time 48.2s
- Epoch 8: train_loss 0.0302, train_psnr 27.5230, val_loss 0.0395, val_psnr 24.0368, ips 67.35, lr 0.000200, time 47.7s
- Epoch 9: train_loss 0.0300, train_psnr 27.5186, val_loss 0.0391, val_psnr 24.0932, ips 67.78, lr 0.000200, time 47.6s
- Epoch 10: train_loss 0.0300, train_psnr 27.5614, val_loss 0.0388, val_psnr 24.1523, ips 67.73, lr 0.000200, time 47.9s
- Epoch 11: train_loss 0.0295, train_psnr 27.6724, val_loss 0.0385, val_psnr 24.2077, ips 66.88, lr 0.000200, time 48.2s
- Epoch 12: train_loss 0.0291, train_psnr 27.8563, val_loss 0.0384, val_psnr 24.2207, ips 67.22, lr 0.000200, time 48.0s
- Epoch 13: train_loss 0.0290, train_psnr 27.9038, val_loss 0.0383, val_psnr 24.2424, ips 67.60, lr 0.000200, time 47.8s
- Epoch 14: train_loss 0.0292, train_psnr 27.8127, val_loss 0.0381, val_psnr 24.2611, ips 66.43, lr 0.000200, time 48.3s
- Epoch 15: train_loss 0.0286, train_psnr 27.8755, val_loss 0.0380, val_psnr 24.2776, ips 67.62, lr 0.000200, time 47.9s
- Epoch 16: train_loss 0.0284, train_psnr 28.0535, val_loss 0.0379, val_psnr 24.2945, ips 67.12, lr 0.000100, time 48.1s
- Epoch 17: train_loss 0.0283, train_psnr 28.0426, val_loss 0.0377, val_psnr 24.3245, ips 66.81, lr 0.000100, time 48.6s
- Epoch 18: train_loss 0.0283, train_psnr 28.0171, val_loss 0.0377, val_psnr 24.3136, ips 65.66, lr 0.000100, time 48.7s
- Epoch 19: train_loss 0.0284, train_psnr 27.9854, val_loss 0.0376, val_psnr 24.3337, ips 68.89, lr 0.000100, time 47.4s
- Epoch 20: train_loss 0.0280, train_psnr 28.0561, val_loss 0.0375, val_psnr 24.3431, ips 67.50, lr 0.000100, time 48.0s
- Epoch 21: train_loss 0.0279, train_psnr 28.1128, val_loss 0.0376, val_psnr 24.3423, ips 66.39, lr 0.000100, time 48.4s
- Epoch 22: train_loss 0.0286, train_psnr 28.0150, val_loss 0.0377, val_psnr 24.3084, ips 68.02, lr 0.000100, time 47.8s
- Epoch 23: train_loss 0.0277, train_psnr 28.0796, val_loss 0.0378, val_psnr 24.3229, ips 68.21, lr 0.000100, time 47.7s
- Epoch 24: train_loss 0.0277, train_psnr 28.1525, val_loss 0.0375, val_psnr 24.3528, ips 68.18, lr 0.000100, time 48.1s

## Qualitative Observations

- The identity baseline is already strong on GoPro, so the first U-Net benchmark needs to clear a fairly high floor.
- The U-Net improves over identity on both validation and test, which confirms that the paired pipeline and restoration model are working end to end.

## Issues Before Stronger Models

- This run uses a modest first-pass U-Net, so model capacity and training duration are still conservative.
- Validation is derived from only three sequences, so conclusions should be treated as directional rather than final.
- The best checkpoint is selected by `val_psnr`, but worker and batch-size settings are still conservative rather than explicitly tuned.
- Full-resolution evaluation is deterministic, but training still samples one random patch per frame per epoch; longer runs should improve stability.
