# Benchmark Report

## Run

- Artifact prefix: `l4-naf-style-20260514-01`

## Train/Validation Split

- Seed: `42`
- Validation fraction from official training sequences: 10%
- Train sequences (19): GOPR0372_07_00, GOPR0372_07_01, GOPR0374_11_01, GOPR0374_11_02, GOPR0378_13_00, GOPR0379_11_00, GOPR0380_11_00, GOPR0384_11_01, GOPR0384_11_02, GOPR0384_11_03, GOPR0384_11_04, GOPR0385_11_00, GOPR0386_11_00, GOPR0477_11_00, GOPR0857_11_00, GOPR0868_11_01, GOPR0868_11_02, GOPR0871_11_01, GOPR0881_11_00
- Validation sequences (3): GOPR0374_11_00, GOPR0374_11_03, GOPR0884_11_00

## Model

- Model key: `naf_style_unet`
- Architecture: `NAFStyleUNet`
- Trainable parameters: 3,029,891
- Total parameters: 3,029,891
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
- NAFStyleUNet validation: PSNR 25.4612, SSIM 0.7724
- NAFStyleUNet test: PSNR 27.3037, SSIM 0.8347
- Validation improvement over identity: PSNR +1.4276, SSIM +0.0374
- Test improvement over identity: PSNR +1.6635, SSIM +0.0417

## Training History

- Epoch 1: train_loss 0.0305, train_psnr 27.5459, val_loss 0.0395, val_psnr 24.0478, ips 24.54, lr 0.000020, time 119.9s
- Epoch 2: train_loss 0.0304, train_psnr 27.5891, val_loss 0.0395, val_psnr 24.0647, ips 30.87, lr 0.000080, time 104.2s
- Epoch 3: train_loss 0.0302, train_psnr 27.6072, val_loss 0.0394, val_psnr 24.0668, ips 36.95, lr 0.000140, time 94.8s
- Epoch 4: train_loss 0.0304, train_psnr 27.4964, val_loss 0.0393, val_psnr 24.0855, ips 40.13, lr 0.000200, time 91.8s
- Epoch 5: train_loss 0.0309, train_psnr 27.3967, val_loss 0.0391, val_psnr 24.1273, ips 40.13, lr 0.000200, time 91.2s
- Epoch 6: train_loss 0.0306, train_psnr 27.4608, val_loss 0.0388, val_psnr 24.1771, ips 40.30, lr 0.000199, time 90.8s
- Epoch 7: train_loss 0.0301, train_psnr 27.5978, val_loss 0.0388, val_psnr 24.1809, ips 40.20, lr 0.000199, time 91.1s
- Epoch 8: train_loss 0.0297, train_psnr 27.6610, val_loss 0.0385, val_psnr 24.2087, ips 39.85, lr 0.000198, time 91.8s
- Epoch 9: train_loss 0.0297, train_psnr 27.6825, val_loss 0.0384, val_psnr 24.2469, ips 40.17, lr 0.000196, time 91.7s
- Epoch 10: train_loss 0.0292, train_psnr 27.8373, val_loss 0.0380, val_psnr 24.3020, ips 40.16, lr 0.000195, time 91.2s
- Epoch 11: train_loss 0.0287, train_psnr 27.9896, val_loss 0.0382, val_psnr 24.2761, ips 40.05, lr 0.000193, time 91.8s
- Epoch 12: train_loss 0.0292, train_psnr 27.7958, val_loss 0.0376, val_psnr 24.3718, ips 39.96, lr 0.000190, time 92.3s
- Epoch 13: train_loss 0.0285, train_psnr 27.9865, val_loss 0.0377, val_psnr 24.3605, ips 39.84, lr 0.000188, time 91.5s
- Epoch 14: train_loss 0.0282, train_psnr 28.0698, val_loss 0.0373, val_psnr 24.4035, ips 39.60, lr 0.000185, time 91.9s
- Epoch 15: train_loss 0.0288, train_psnr 27.8999, val_loss 0.0369, val_psnr 24.4438, ips 38.32, lr 0.000182, time 94.5s
- Epoch 16: train_loss 0.0273, train_psnr 28.2442, val_loss 0.0369, val_psnr 24.4887, ips 39.69, lr 0.000179, time 93.1s
- Epoch 17: train_loss 0.0275, train_psnr 28.2237, val_loss 0.0370, val_psnr 24.3712, ips 40.18, lr 0.000176, time 91.9s
- Epoch 18: train_loss 0.0277, train_psnr 28.1902, val_loss 0.0367, val_psnr 24.5346, ips 40.00, lr 0.000172, time 92.2s
- Epoch 19: train_loss 0.0269, train_psnr 28.3921, val_loss 0.0367, val_psnr 24.5256, ips 39.75, lr 0.000168, time 92.3s
- Epoch 20: train_loss 0.0272, train_psnr 28.3152, val_loss 0.0362, val_psnr 24.6276, ips 39.97, lr 0.000164, time 91.7s
- Epoch 21: train_loss 0.0268, train_psnr 28.3589, val_loss 0.0362, val_psnr 24.6304, ips 40.04, lr 0.000159, time 91.5s
- Epoch 22: train_loss 0.0268, train_psnr 28.4076, val_loss 0.0358, val_psnr 24.6761, ips 39.75, lr 0.000155, time 92.2s
- Epoch 23: train_loss 0.0267, train_psnr 28.3896, val_loss 0.0359, val_psnr 24.6889, ips 40.05, lr 0.000150, time 91.6s
- Epoch 24: train_loss 0.0264, train_psnr 28.4766, val_loss 0.0358, val_psnr 24.6410, ips 40.01, lr 0.000145, time 91.5s
- Epoch 25: train_loss 0.0261, train_psnr 28.6026, val_loss 0.0355, val_psnr 24.7169, ips 39.90, lr 0.000140, time 91.7s
- Epoch 26: train_loss 0.0261, train_psnr 28.5553, val_loss 0.0353, val_psnr 24.8206, ips 39.87, lr 0.000135, time 91.7s
- Epoch 27: train_loss 0.0261, train_psnr 28.6183, val_loss 0.0350, val_psnr 24.8621, ips 40.09, lr 0.000130, time 92.3s
- Epoch 28: train_loss 0.0252, train_psnr 28.9230, val_loss 0.0348, val_psnr 24.9214, ips 39.69, lr 0.000125, time 92.0s
- Epoch 29: train_loss 0.0258, train_psnr 28.6361, val_loss 0.0347, val_psnr 24.9599, ips 39.33, lr 0.000120, time 92.4s
- Epoch 30: train_loss 0.0254, train_psnr 28.7518, val_loss 0.0349, val_psnr 24.8894, ips 39.53, lr 0.000114, time 92.5s
- Epoch 31: train_loss 0.0253, train_psnr 28.7778, val_loss 0.0349, val_psnr 24.8917, ips 39.86, lr 0.000109, time 93.0s
- Epoch 32: train_loss 0.0251, train_psnr 28.8863, val_loss 0.0340, val_psnr 25.0961, ips 39.91, lr 0.000103, time 91.4s
- Epoch 33: train_loss 0.0251, train_psnr 28.8103, val_loss 0.0341, val_psnr 25.0892, ips 40.03, lr 0.000098, time 91.6s
- Epoch 34: train_loss 0.0251, train_psnr 28.8537, val_loss 0.0341, val_psnr 25.0875, ips 39.57, lr 0.000092, time 92.7s
- Epoch 35: train_loss 0.0249, train_psnr 28.9289, val_loss 0.0339, val_psnr 25.1196, ips 39.85, lr 0.000087, time 91.6s
- Epoch 36: train_loss 0.0244, train_psnr 29.0494, val_loss 0.0339, val_psnr 25.1254, ips 39.98, lr 0.000081, time 91.8s
- Epoch 37: train_loss 0.0251, train_psnr 28.9057, val_loss 0.0335, val_psnr 25.2078, ips 40.08, lr 0.000076, time 91.6s
- Epoch 38: train_loss 0.0246, train_psnr 28.9680, val_loss 0.0335, val_psnr 25.2143, ips 40.17, lr 0.000071, time 91.5s
- Epoch 39: train_loss 0.0242, train_psnr 29.1639, val_loss 0.0333, val_psnr 25.2622, ips 40.05, lr 0.000066, time 92.0s
- Epoch 40: train_loss 0.0240, train_psnr 29.2103, val_loss 0.0334, val_psnr 25.2487, ips 40.02, lr 0.000061, time 91.8s
- Epoch 41: train_loss 0.0247, train_psnr 29.0051, val_loss 0.0333, val_psnr 25.2813, ips 39.76, lr 0.000056, time 92.2s
- Epoch 42: train_loss 0.0241, train_psnr 29.2620, val_loss 0.0334, val_psnr 25.2354, ips 39.08, lr 0.000051, time 93.1s
- Epoch 43: train_loss 0.0236, train_psnr 29.4037, val_loss 0.0332, val_psnr 25.3022, ips 39.69, lr 0.000046, time 91.8s
- Epoch 44: train_loss 0.0242, train_psnr 29.2416, val_loss 0.0330, val_psnr 25.3319, ips 39.98, lr 0.000042, time 91.4s
- Epoch 45: train_loss 0.0241, train_psnr 29.1766, val_loss 0.0329, val_psnr 25.3667, ips 39.85, lr 0.000037, time 92.9s
- Epoch 46: train_loss 0.0241, train_psnr 29.2504, val_loss 0.0329, val_psnr 25.3725, ips 39.90, lr 0.000033, time 91.6s
- Epoch 47: train_loss 0.0236, train_psnr 29.4044, val_loss 0.0329, val_psnr 25.3748, ips 39.82, lr 0.000029, time 92.0s
- Epoch 48: train_loss 0.0236, train_psnr 29.3131, val_loss 0.0328, val_psnr 25.3941, ips 39.74, lr 0.000025, time 91.9s
- Epoch 49: train_loss 0.0239, train_psnr 29.2565, val_loss 0.0327, val_psnr 25.3956, ips 40.15, lr 0.000022, time 91.2s
- Epoch 50: train_loss 0.0238, train_psnr 29.3150, val_loss 0.0328, val_psnr 25.3934, ips 40.12, lr 0.000019, time 92.0s
- Epoch 51: train_loss 0.0236, train_psnr 29.3809, val_loss 0.0327, val_psnr 25.4207, ips 39.86, lr 0.000016, time 92.6s
- Epoch 52: train_loss 0.0234, train_psnr 29.4241, val_loss 0.0326, val_psnr 25.4458, ips 40.12, lr 0.000013, time 91.3s
- Epoch 53: train_loss 0.0234, train_psnr 29.4398, val_loss 0.0326, val_psnr 25.4533, ips 39.77, lr 0.000011, time 92.2s
- Epoch 54: train_loss 0.0236, train_psnr 29.3430, val_loss 0.0326, val_psnr 25.4459, ips 39.95, lr 0.000008, time 92.1s
- Epoch 55: train_loss 0.0237, train_psnr 29.3067, val_loss 0.0326, val_psnr 25.4570, ips 39.93, lr 0.000006, time 92.1s
- Epoch 56: train_loss 0.0238, train_psnr 29.2384, val_loss 0.0325, val_psnr 25.4586, ips 39.99, lr 0.000005, time 91.6s
- Epoch 57: train_loss 0.0235, train_psnr 29.3710, val_loss 0.0325, val_psnr 25.4565, ips 40.06, lr 0.000003, time 92.4s
- Epoch 58: train_loss 0.0235, train_psnr 29.3587, val_loss 0.0325, val_psnr 25.4589, ips 39.88, lr 0.000002, time 92.5s
- Epoch 59: train_loss 0.0234, train_psnr 29.4641, val_loss 0.0325, val_psnr 25.4553, ips 39.94, lr 0.000002, time 92.3s
- Epoch 60: train_loss 0.0234, train_psnr 29.3890, val_loss 0.0325, val_psnr 25.4612, ips 39.63, lr 0.000001, time 92.6s

## Qualitative Observations

- The identity baseline is already strong on GoPro, but this custom NAF-style U-Net clears that floor decisively: `+1.4276 dB` PSNR on validation and `+1.6635 dB` on test over identity.
- Relative to the previous best plain residual U-Net run (`l4-charb-cosine-20260512-01`), this run improved by `+0.8058 dB` validation PSNR, `+0.7381 dB` test PSNR, `+0.0221` validation SSIM, and `+0.0208` test SSIM.
- Validation PSNR continued rising through epoch `60`, so the `Charbonnier + warmup/cosine` recipe remained productive for this architecture instead of plateauing early.

## Issues Before Stronger Models

- Validation still comes from only three held-out sequences, so the win is meaningful but not yet a substitute for a broader validation protocol.
- This model is a custom `NAFStyleUNet`, not a faithful published `NAFNet-lite` baseline, so there is still no direct literature-style calibration point in the project.
- Training still uses `256x256` random patches with one sampled crop per frame per epoch, so long-range motion context remains partially constrained during learning.
- Despite the lower parameter count than the plain residual U-Net, throughput was lower, so block efficiency is still a practical tradeoff to revisit if training cost becomes a bottleneck.
