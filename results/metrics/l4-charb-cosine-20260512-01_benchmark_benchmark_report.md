# Benchmark Report

## Run

- Artifact prefix: `l4-charb-cosine-20260512-01`

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
- Scheduler: SequentialLR(LinearLR(start_factor=0.1, total_iters=3), CosineAnnealingLR(T_max=57, eta_min=1e-06), milestones=[3])
- Loss: Charbonnier(eps=0.001)

## Metrics

- Identity validation: PSNR 24.0336, SSIM 0.7350
- Identity test: PSNR 25.6401, SSIM 0.7930
- U-Net validation: PSNR 24.6554, SSIM 0.7504
- U-Net test: PSNR 26.5656, SSIM 0.8139
- Validation improvement over identity: PSNR +0.6218, SSIM +0.0154
- Test improvement over identity: PSNR +0.9255, SSIM +0.0209

## Training History

- Epoch 1: train_loss 0.0308, train_psnr 27.3263, val_loss 0.0396, val_psnr 24.0384, ips 5.59, lr 0.000020, time 337.5s
- Epoch 2: train_loss 0.0307, train_psnr 27.4056, val_loss 0.0395, val_psnr 24.0436, ips 50.34, lr 0.000080, time 52.7s
- Epoch 3: train_loss 0.0305, train_psnr 27.4901, val_loss 0.0394, val_psnr 24.0651, ips 58.64, lr 0.000140, time 46.2s
- Epoch 4: train_loss 0.0301, train_psnr 27.6636, val_loss 0.0394, val_psnr 24.0712, ips 60.38, lr 0.000200, time 50.5s
- Epoch 5: train_loss 0.0308, train_psnr 27.4085, val_loss 0.0393, val_psnr 24.0874, ips 62.25, lr 0.000200, time 43.4s
- Epoch 6: train_loss 0.0307, train_psnr 27.4294, val_loss 0.0391, val_psnr 24.1183, ips 60.93, lr 0.000199, time 46.4s
- Epoch 7: train_loss 0.0302, train_psnr 27.5657, val_loss 0.0389, val_psnr 24.1491, ips 61.31, lr 0.000199, time 48.7s
- Epoch 8: train_loss 0.0298, train_psnr 27.6757, val_loss 0.0387, val_psnr 24.1792, ips 60.92, lr 0.000198, time 49.7s
- Epoch 9: train_loss 0.0298, train_psnr 27.6004, val_loss 0.0388, val_psnr 24.1618, ips 60.60, lr 0.000196, time 47.2s
- Epoch 10: train_loss 0.0296, train_psnr 27.6883, val_loss 0.0384, val_psnr 24.2276, ips 59.50, lr 0.000195, time 45.0s
- Epoch 11: train_loss 0.0296, train_psnr 27.6417, val_loss 0.0384, val_psnr 24.2335, ips 60.44, lr 0.000193, time 47.9s
- Epoch 12: train_loss 0.0294, train_psnr 27.7814, val_loss 0.0384, val_psnr 24.2483, ips 60.57, lr 0.000190, time 49.7s
- Epoch 13: train_loss 0.0290, train_psnr 27.8516, val_loss 0.0382, val_psnr 24.2475, ips 59.33, lr 0.000188, time 49.9s
- Epoch 14: train_loss 0.0292, train_psnr 27.8272, val_loss 0.0380, val_psnr 24.2910, ips 59.37, lr 0.000185, time 48.8s
- Epoch 15: train_loss 0.0286, train_psnr 27.9418, val_loss 0.0380, val_psnr 24.3054, ips 60.73, lr 0.000182, time 47.0s
- Epoch 16: train_loss 0.0286, train_psnr 27.9522, val_loss 0.0378, val_psnr 24.3045, ips 59.60, lr 0.000179, time 45.0s
- Epoch 17: train_loss 0.0285, train_psnr 28.0008, val_loss 0.0379, val_psnr 24.3181, ips 59.55, lr 0.000176, time 46.3s
- Epoch 18: train_loss 0.0285, train_psnr 28.0355, val_loss 0.0378, val_psnr 24.3570, ips 58.19, lr 0.000172, time 46.2s
- Epoch 19: train_loss 0.0280, train_psnr 28.1505, val_loss 0.0378, val_psnr 24.3296, ips 57.46, lr 0.000168, time 50.2s
- Epoch 20: train_loss 0.0285, train_psnr 28.0074, val_loss 0.0376, val_psnr 24.3121, ips 58.76, lr 0.000164, time 46.5s
- Epoch 21: train_loss 0.0280, train_psnr 28.0178, val_loss 0.0374, val_psnr 24.3911, ips 58.51, lr 0.000159, time 47.4s
- Epoch 22: train_loss 0.0279, train_psnr 28.0576, val_loss 0.0374, val_psnr 24.3684, ips 59.24, lr 0.000155, time 47.3s
- Epoch 23: train_loss 0.0282, train_psnr 28.1243, val_loss 0.0374, val_psnr 24.3946, ips 59.57, lr 0.000150, time 49.1s
- Epoch 24: train_loss 0.0282, train_psnr 28.1079, val_loss 0.0372, val_psnr 24.4079, ips 60.13, lr 0.000145, time 47.5s
- Epoch 25: train_loss 0.0280, train_psnr 28.1502, val_loss 0.0374, val_psnr 24.4104, ips 58.10, lr 0.000140, time 51.9s
- Epoch 26: train_loss 0.0277, train_psnr 28.2017, val_loss 0.0373, val_psnr 24.3883, ips 58.88, lr 0.000135, time 45.9s
- Epoch 27: train_loss 0.0274, train_psnr 28.2343, val_loss 0.0371, val_psnr 24.4572, ips 59.07, lr 0.000130, time 44.9s
- Epoch 28: train_loss 0.0279, train_psnr 28.0968, val_loss 0.0374, val_psnr 24.4137, ips 59.14, lr 0.000125, time 48.4s
- Epoch 29: train_loss 0.0277, train_psnr 28.1238, val_loss 0.0368, val_psnr 24.4790, ips 59.03, lr 0.000120, time 46.9s
- Epoch 30: train_loss 0.0272, train_psnr 28.3440, val_loss 0.0369, val_psnr 24.4629, ips 58.04, lr 0.000114, time 45.6s
- Epoch 31: train_loss 0.0271, train_psnr 28.3216, val_loss 0.0369, val_psnr 24.4495, ips 57.19, lr 0.000109, time 49.0s
- Epoch 32: train_loss 0.0271, train_psnr 28.3452, val_loss 0.0367, val_psnr 24.5117, ips 59.83, lr 0.000103, time 48.2s
- Epoch 33: train_loss 0.0274, train_psnr 28.2016, val_loss 0.0368, val_psnr 24.4496, ips 58.42, lr 0.000098, time 48.4s
- Epoch 34: train_loss 0.0274, train_psnr 28.2238, val_loss 0.0366, val_psnr 24.5254, ips 58.92, lr 0.000092, time 45.7s
- Epoch 35: train_loss 0.0267, train_psnr 28.4741, val_loss 0.0365, val_psnr 24.5470, ips 58.64, lr 0.000087, time 48.0s
- Epoch 36: train_loss 0.0272, train_psnr 28.3351, val_loss 0.0365, val_psnr 24.5593, ips 59.91, lr 0.000081, time 47.1s
- Epoch 37: train_loss 0.0273, train_psnr 28.3662, val_loss 0.0365, val_psnr 24.5368, ips 59.42, lr 0.000076, time 45.8s
- Epoch 38: train_loss 0.0269, train_psnr 28.3945, val_loss 0.0364, val_psnr 24.5555, ips 60.24, lr 0.000071, time 46.9s
- Epoch 39: train_loss 0.0270, train_psnr 28.4193, val_loss 0.0364, val_psnr 24.5932, ips 60.22, lr 0.000066, time 48.1s
- Epoch 40: train_loss 0.0269, train_psnr 28.3586, val_loss 0.0365, val_psnr 24.5710, ips 59.51, lr 0.000061, time 47.1s
- Epoch 41: train_loss 0.0268, train_psnr 28.4675, val_loss 0.0364, val_psnr 24.5783, ips 59.07, lr 0.000056, time 49.9s
- Epoch 42: train_loss 0.0269, train_psnr 28.4359, val_loss 0.0363, val_psnr 24.5863, ips 60.55, lr 0.000051, time 46.7s
- Epoch 43: train_loss 0.0270, train_psnr 28.3448, val_loss 0.0363, val_psnr 24.5952, ips 58.89, lr 0.000046, time 47.2s
- Epoch 44: train_loss 0.0269, train_psnr 28.3117, val_loss 0.0361, val_psnr 24.6181, ips 59.03, lr 0.000042, time 49.7s
- Epoch 45: train_loss 0.0267, train_psnr 28.4200, val_loss 0.0362, val_psnr 24.6229, ips 60.19, lr 0.000037, time 46.3s
- Epoch 46: train_loss 0.0264, train_psnr 28.5943, val_loss 0.0361, val_psnr 24.6230, ips 59.39, lr 0.000033, time 47.4s
- Epoch 47: train_loss 0.0270, train_psnr 28.2580, val_loss 0.0361, val_psnr 24.6341, ips 59.90, lr 0.000029, time 48.1s
- Epoch 48: train_loss 0.0261, train_psnr 28.6240, val_loss 0.0361, val_psnr 24.6174, ips 59.12, lr 0.000025, time 51.6s
- Epoch 49: train_loss 0.0265, train_psnr 28.4269, val_loss 0.0361, val_psnr 24.6311, ips 59.34, lr 0.000022, time 49.9s
- Epoch 50: train_loss 0.0265, train_psnr 28.5099, val_loss 0.0360, val_psnr 24.6360, ips 59.57, lr 0.000019, time 47.5s
- Epoch 51: train_loss 0.0269, train_psnr 28.4172, val_loss 0.0360, val_psnr 24.6367, ips 58.57, lr 0.000016, time 47.3s
- Epoch 52: train_loss 0.0267, train_psnr 28.4418, val_loss 0.0360, val_psnr 24.6406, ips 58.55, lr 0.000013, time 48.5s
- Epoch 53: train_loss 0.0266, train_psnr 28.4576, val_loss 0.0360, val_psnr 24.6545, ips 59.00, lr 0.000011, time 50.6s
- Epoch 54: train_loss 0.0263, train_psnr 28.5463, val_loss 0.0360, val_psnr 24.6480, ips 57.49, lr 0.000008, time 46.7s
- Epoch 55: train_loss 0.0270, train_psnr 28.3640, val_loss 0.0359, val_psnr 24.6495, ips 58.34, lr 0.000006, time 46.1s
- Epoch 56: train_loss 0.0262, train_psnr 28.5459, val_loss 0.0359, val_psnr 24.6553, ips 57.66, lr 0.000005, time 48.6s
- Epoch 57: train_loss 0.0266, train_psnr 28.4727, val_loss 0.0359, val_psnr 24.6493, ips 59.41, lr 0.000003, time 48.6s
- Epoch 58: train_loss 0.0268, train_psnr 28.4879, val_loss 0.0359, val_psnr 24.6495, ips 60.12, lr 0.000002, time 48.0s
- Epoch 59: train_loss 0.0263, train_psnr 28.5927, val_loss 0.0359, val_psnr 24.6543, ips 58.52, lr 0.000002, time 47.2s
- Epoch 60: train_loss 0.0260, train_psnr 28.6703, val_loss 0.0359, val_psnr 24.6554, ips 58.40, lr 0.000001, time 49.0s

## Qualitative Observations

- The identity baseline is already strong on GoPro, so a plain U-Net baseline needs to clear a fairly high floor.
- The U-Net improves over identity on both validation and test, which confirms that the paired pipeline and restoration model are working end to end.

## Issues Before Stronger Models

- This run uses a modest U-Net baseline, so model capacity and training duration are still conservative.
- Validation is derived from only three sequences, so conclusions should be treated as directional rather than final.
- The best checkpoint is selected by `val_psnr`, but worker and batch-size settings are still conservative rather than explicitly tuned.
- Full-resolution evaluation is deterministic, but training still samples one random patch per frame per epoch; longer runs should improve stability.
