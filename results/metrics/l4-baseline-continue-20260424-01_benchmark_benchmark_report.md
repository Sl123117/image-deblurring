# First Benchmark Report

## Run

- Artifact prefix: `l4-baseline-continue-20260424-01`

## Resume

- Resumed from checkpoint: `/root/project/results/checkpoints/l4-baseline-20260421-01_benchmark_unet_deblurring_latest.pt`
- Resume start epoch: 24
- History restored from: `metrics`
- Restored epochs in history: 24
- AMP scaler state restored: False

## Train/Validation Split

- Seed: `42`
- Validation fraction from official training sequences: 10%
- Train sequences (19): GOPR0372_07_00, GOPR0372_07_01, GOPR0374_11_01, GOPR0374_11_02, GOPR0378_13_00, GOPR0379_11_00, GOPR0380_11_00, GOPR0384_11_01, GOPR0384_11_02, GOPR0384_11_03, GOPR0384_11_04, GOPR0385_11_00, GOPR0386_11_00, GOPR0477_11_00, GOPR0857_11_00, GOPR0868_11_01, GOPR0868_11_02, GOPR0871_11_01, GOPR0881_11_00
- Validation sequences (3): GOPR0374_11_00, GOPR0374_11_03, GOPR0884_11_00

## Model

- Architecture: `UNetDeblurModel`
- Trainable parameters: 3,348,355
- Total parameters: 3,348,355
- Base channels: 32
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
- Epochs: 60
- Validate each epoch: True
- Checkpoint monitor: val_psnr
- Optimizer: Adam
- Learning rate: 0.0002
- Scheduler: StepLR(step_size=15, gamma=0.5)
- Loss: L1

## Metrics

- Identity validation: PSNR 24.0336, SSIM 0.7350
- Identity test: PSNR 25.6401, SSIM 0.7930
- U-Net validation: PSNR 24.4627, SSIM 0.7456
- U-Net test: PSNR 26.3940, SSIM 0.8101
- Validation improvement over identity: PSNR +0.4291, SSIM +0.0106
- Test improvement over identity: PSNR +0.7539, SSIM +0.0171

## Training History

- Epoch 1: train_loss 0.0307, train_psnr 27.3244, val_loss 0.0395, val_psnr 24.0341, ips 5.20, lr 0.000200, time 360.0s
- Epoch 2: train_loss 0.0306, train_psnr 27.3967, val_loss 0.0395, val_psnr 24.0386, ips 54.31, lr 0.000200, time 46.1s
- Epoch 3: train_loss 0.0304, train_psnr 27.4996, val_loss 0.0393, val_psnr 24.0644, ips 66.37, lr 0.000200, time 40.3s
- Epoch 4: train_loss 0.0299, train_psnr 27.6980, val_loss 0.0393, val_psnr 24.0738, ips 66.36, lr 0.000200, time 41.2s
- Epoch 5: train_loss 0.0306, train_psnr 27.4315, val_loss 0.0392, val_psnr 24.0923, ips 63.59, lr 0.000200, time 42.2s
- Epoch 6: train_loss 0.0306, train_psnr 27.4377, val_loss 0.0392, val_psnr 24.0961, ips 62.94, lr 0.000200, time 42.8s
- Epoch 7: train_loss 0.0304, train_psnr 27.4918, val_loss 0.0391, val_psnr 24.1064, ips 66.23, lr 0.000200, time 40.2s
- Epoch 8: train_loss 0.0300, train_psnr 27.6015, val_loss 0.0390, val_psnr 24.1161, ips 63.74, lr 0.000200, time 41.7s
- Epoch 9: train_loss 0.0300, train_psnr 27.5366, val_loss 0.0389, val_psnr 24.1432, ips 66.02, lr 0.000200, time 40.9s
- Epoch 10: train_loss 0.0296, train_psnr 27.6435, val_loss 0.0384, val_psnr 24.2100, ips 66.05, lr 0.000200, time 41.1s
- Epoch 11: train_loss 0.0296, train_psnr 27.6067, val_loss 0.0384, val_psnr 24.2117, ips 63.90, lr 0.000200, time 42.5s
- Epoch 12: train_loss 0.0294, train_psnr 27.7481, val_loss 0.0390, val_psnr 24.1543, ips 64.78, lr 0.000200, time 41.8s
- Epoch 13: train_loss 0.0292, train_psnr 27.7872, val_loss 0.0382, val_psnr 24.2380, ips 62.93, lr 0.000200, time 42.3s
- Epoch 14: train_loss 0.0293, train_psnr 27.7671, val_loss 0.0381, val_psnr 24.2647, ips 66.47, lr 0.000200, time 40.3s
- Epoch 15: train_loss 0.0289, train_psnr 27.8383, val_loss 0.0383, val_psnr 24.2335, ips 66.25, lr 0.000200, time 40.3s
- Epoch 16: train_loss 0.0288, train_psnr 27.8539, val_loss 0.0382, val_psnr 24.2544, ips 62.75, lr 0.000100, time 42.5s
- Epoch 17: train_loss 0.0287, train_psnr 27.9207, val_loss 0.0381, val_psnr 24.2573, ips 66.31, lr 0.000100, time 41.7s
- Epoch 18: train_loss 0.0287, train_psnr 27.9484, val_loss 0.0379, val_psnr 24.2911, ips 66.09, lr 0.000100, time 42.2s
- Epoch 19: train_loss 0.0283, train_psnr 28.0538, val_loss 0.0380, val_psnr 24.2751, ips 66.56, lr 0.000100, time 40.1s
- Epoch 20: train_loss 0.0288, train_psnr 27.9092, val_loss 0.0379, val_psnr 24.2871, ips 64.53, lr 0.000100, time 41.8s
- Epoch 21: train_loss 0.0282, train_psnr 27.9224, val_loss 0.0378, val_psnr 24.3061, ips 65.43, lr 0.000100, time 40.8s
- Epoch 22: train_loss 0.0282, train_psnr 27.9647, val_loss 0.0377, val_psnr 24.3165, ips 66.87, lr 0.000100, time 41.8s
- Epoch 23: train_loss 0.0285, train_psnr 28.0174, val_loss 0.0378, val_psnr 24.3247, ips 65.63, lr 0.000100, time 41.5s
- Epoch 24: train_loss 0.0286, train_psnr 27.9857, val_loss 0.0376, val_psnr 24.3450, ips 66.44, lr 0.000100, time 40.8s
- Epoch 25: train_loss 0.0282, train_psnr 27.9713, val_loss 0.0376, val_psnr 24.3555, ips 59.34, lr 0.000100, time 47.3s
- Epoch 26: train_loss 0.0281, train_psnr 28.0495, val_loss 0.0376, val_psnr 24.3381, ips 59.53, lr 0.000100, time 44.4s
- Epoch 27: train_loss 0.0279, train_psnr 28.1146, val_loss 0.0375, val_psnr 24.3628, ips 60.21, lr 0.000100, time 46.4s
- Epoch 28: train_loss 0.0277, train_psnr 28.2283, val_loss 0.0376, val_psnr 24.3468, ips 63.87, lr 0.000100, time 41.1s
- Epoch 29: train_loss 0.0283, train_psnr 27.9538, val_loss 0.0375, val_psnr 24.3598, ips 64.11, lr 0.000100, time 41.8s
- Epoch 30: train_loss 0.0284, train_psnr 27.9168, val_loss 0.0374, val_psnr 24.3834, ips 64.46, lr 0.000100, time 42.3s
- Epoch 31: train_loss 0.0281, train_psnr 28.0184, val_loss 0.0373, val_psnr 24.3816, ips 63.46, lr 0.000050, time 43.0s
- Epoch 32: train_loss 0.0278, train_psnr 28.0979, val_loss 0.0374, val_psnr 24.3633, ips 63.01, lr 0.000050, time 42.4s
- Epoch 33: train_loss 0.0279, train_psnr 27.9967, val_loss 0.0373, val_psnr 24.3859, ips 63.34, lr 0.000050, time 42.8s
- Epoch 34: train_loss 0.0279, train_psnr 28.0344, val_loss 0.0373, val_psnr 24.4011, ips 63.96, lr 0.000050, time 42.3s
- Epoch 35: train_loss 0.0280, train_psnr 27.9722, val_loss 0.0373, val_psnr 24.3813, ips 63.95, lr 0.000050, time 42.2s
- Epoch 36: train_loss 0.0280, train_psnr 28.0872, val_loss 0.0372, val_psnr 24.3990, ips 64.51, lr 0.000050, time 42.3s
- Epoch 37: train_loss 0.0277, train_psnr 28.1291, val_loss 0.0371, val_psnr 24.4144, ips 64.97, lr 0.000050, time 42.3s
- Epoch 38: train_loss 0.0280, train_psnr 28.0835, val_loss 0.0372, val_psnr 24.3919, ips 64.04, lr 0.000050, time 44.0s
- Epoch 39: train_loss 0.0274, train_psnr 28.1817, val_loss 0.0371, val_psnr 24.4129, ips 64.00, lr 0.000050, time 45.6s
- Epoch 40: train_loss 0.0274, train_psnr 28.1852, val_loss 0.0371, val_psnr 24.3951, ips 62.53, lr 0.000050, time 44.2s
- Epoch 41: train_loss 0.0275, train_psnr 28.2096, val_loss 0.0371, val_psnr 24.4194, ips 63.34, lr 0.000050, time 44.2s
- Epoch 42: train_loss 0.0275, train_psnr 28.2211, val_loss 0.0371, val_psnr 24.4291, ips 63.87, lr 0.000050, time 43.6s
- Epoch 43: train_loss 0.0271, train_psnr 28.3254, val_loss 0.0372, val_psnr 24.3932, ips 63.71, lr 0.000050, time 41.6s
- Epoch 44: train_loss 0.0277, train_psnr 28.1628, val_loss 0.0370, val_psnr 24.4230, ips 63.13, lr 0.000050, time 44.0s
- Epoch 45: train_loss 0.0272, train_psnr 28.1566, val_loss 0.0371, val_psnr 24.3999, ips 65.18, lr 0.000050, time 40.6s
- Epoch 46: train_loss 0.0272, train_psnr 28.2003, val_loss 0.0370, val_psnr 24.4312, ips 63.22, lr 0.000025, time 41.8s
- Epoch 47: train_loss 0.0275, train_psnr 28.2611, val_loss 0.0369, val_psnr 24.4416, ips 64.26, lr 0.000025, time 43.4s
- Epoch 48: train_loss 0.0276, train_psnr 28.2216, val_loss 0.0370, val_psnr 24.4133, ips 64.24, lr 0.000025, time 41.3s
- Epoch 49: train_loss 0.0274, train_psnr 28.2473, val_loss 0.0369, val_psnr 24.4344, ips 63.36, lr 0.000025, time 42.9s
- Epoch 50: train_loss 0.0271, train_psnr 28.3087, val_loss 0.0369, val_psnr 24.4531, ips 63.36, lr 0.000025, time 41.6s
- Epoch 51: train_loss 0.0270, train_psnr 28.3172, val_loss 0.0368, val_psnr 24.4545, ips 63.67, lr 0.000025, time 44.5s
- Epoch 52: train_loss 0.0275, train_psnr 28.1603, val_loss 0.0368, val_psnr 24.4595, ips 63.80, lr 0.000025, time 42.4s
- Epoch 53: train_loss 0.0274, train_psnr 28.1714, val_loss 0.0369, val_psnr 24.4503, ips 64.04, lr 0.000025, time 41.2s
- Epoch 54: train_loss 0.0269, train_psnr 28.3716, val_loss 0.0368, val_psnr 24.4619, ips 65.68, lr 0.000025, time 42.4s
- Epoch 55: train_loss 0.0269, train_psnr 28.3513, val_loss 0.0368, val_psnr 24.4372, ips 65.19, lr 0.000025, time 41.5s
- Epoch 56: train_loss 0.0270, train_psnr 28.3527, val_loss 0.0367, val_psnr 24.4627, ips 65.40, lr 0.000025, time 42.0s
- Epoch 57: train_loss 0.0273, train_psnr 28.1945, val_loss 0.0368, val_psnr 24.4612, ips 65.18, lr 0.000025, time 42.0s
- Epoch 58: train_loss 0.0273, train_psnr 28.2024, val_loss 0.0368, val_psnr 24.4440, ips 65.29, lr 0.000025, time 40.8s
- Epoch 59: train_loss 0.0266, train_psnr 28.4492, val_loss 0.0367, val_psnr 24.4586, ips 65.22, lr 0.000025, time 40.9s
- Epoch 60: train_loss 0.0273, train_psnr 28.2996, val_loss 0.0367, val_psnr 24.4618, ips 65.09, lr 0.000025, time 40.8s

## Qualitative Observations

- The identity baseline is already strong on GoPro, so the first U-Net benchmark needs to clear a fairly high floor.
- The U-Net improves over identity on both validation and test, which confirms that the paired pipeline and restoration model are working end to end.

## Issues Before Stronger Models

- This run uses a modest first-pass U-Net, so model capacity and training duration are still conservative.
- Validation is derived from only three sequences, so conclusions should be treated as directional rather than final.
- The best checkpoint is selected by `val_psnr`, but worker and batch-size settings are still conservative rather than explicitly tuned.
- Full-resolution evaluation is deterministic, but training still samples one random patch per frame per epoch; longer runs should improve stability.
