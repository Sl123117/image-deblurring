# Benchmark Report

## Run

- Artifact prefix: `l4-nafnet-lite-continue-20260517-01`

## Checkpoint Evaluation

- Requested checkpoint: `/root/project/results/checkpoints/l4-nafnet-lite-continue-20260517-01_benchmark_nafnet_lite_baseline_deblurring_latest.pt`
- Best checkpoint evaluated: `/root/project/results/checkpoints/l4-nafnet-lite-continue-20260517-01_benchmark_nafnet_lite_baseline_deblurring_best.pt`
- Latest checkpoint used for history: `/root/project/results/checkpoints/l4-nafnet-lite-continue-20260517-01_benchmark_nafnet_lite_baseline_deblurring_latest.pt`
- Restored epochs in history: 53

## Train/Validation Split

- Seed: `42`
- Validation fraction from official training sequences: 10%
- Train sequences (19): GOPR0372_07_00, GOPR0372_07_01, GOPR0374_11_01, GOPR0374_11_02, GOPR0378_13_00, GOPR0379_11_00, GOPR0380_11_00, GOPR0384_11_01, GOPR0384_11_02, GOPR0384_11_03, GOPR0384_11_04, GOPR0385_11_00, GOPR0386_11_00, GOPR0477_11_00, GOPR0857_11_00, GOPR0868_11_01, GOPR0868_11_02, GOPR0871_11_01, GOPR0881_11_00
- Validation sequences (3): GOPR0374_11_00, GOPR0374_11_03, GOPR0884_11_00

## Model

- Model key: `nafnet_lite_baseline`
- Architecture: `NAFNetLiteBaseline`
- Trainable parameters: 3,249,997
- Total parameters: 3,249,997
- Base channels / width: 22
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
- Completed epochs recorded: 53
- Validate each epoch: True
- Checkpoint monitor: val_psnr
- Optimizer: Adam
- Learning rate: 0.0002
- Scheduler: SequentialLR(LinearLR(start_factor=0.1, total_iters=3), CosineAnnealingLR(T_max=57, eta_min=1e-06), milestones=[3])
- Loss: Charbonnier(eps=0.001)

## Metrics

- Identity validation: PSNR 24.0336, SSIM 0.7350
- Identity test: PSNR 25.6401, SSIM 0.7930
- NAFNetLiteBaseline validation: PSNR 24.7910, SSIM 0.7515
- NAFNetLiteBaseline test: PSNR 26.3959, SSIM 0.8104
- Validation improvement over identity: PSNR +0.7574, SSIM +0.0165
- Test improvement over identity: PSNR +0.7557, SSIM +0.0175

## Training History

- Epoch 1: train_loss 0.0678, train_psnr 21.2990, val_loss 0.0527, val_psnr 22.5310, ips 38.28, lr 0.000020, time 84.6s
- Epoch 2: train_loss 0.0380, train_psnr 25.0380, val_loss 0.0410, val_psnr 23.7286, ips 43.76, lr 0.000080, time 78.8s
- Epoch 3: train_loss 0.0337, train_psnr 25.8044, val_loss 0.0401, val_psnr 23.8926, ips 48.20, lr 0.000140, time 75.5s
- Epoch 4: train_loss 0.0319, train_psnr 26.6989, val_loss 0.0397, val_psnr 24.0210, ips 49.23, lr 0.000200, time 74.8s
- Epoch 5: train_loss 0.0310, train_psnr 27.2249, val_loss 0.0395, val_psnr 24.0518, ips 49.29, lr 0.000200, time 74.6s
- Epoch 6: train_loss 0.0314, train_psnr 27.1938, val_loss 0.0395, val_psnr 24.0646, ips 49.12, lr 0.000199, time 74.9s
- Epoch 7: train_loss 0.0301, train_psnr 27.6272, val_loss 0.0394, val_psnr 24.0723, ips 48.63, lr 0.000199, time 75.5s
- Epoch 8: train_loss 0.0306, train_psnr 27.4626, val_loss 0.0393, val_psnr 24.0934, ips 47.40, lr 0.000198, time 76.5s
- Epoch 9: train_loss 0.0303, train_psnr 27.5628, val_loss 0.0392, val_psnr 24.1096, ips 49.54, lr 0.000196, time 75.7s
- Epoch 10: train_loss 0.0302, train_psnr 27.5807, val_loss 0.0391, val_psnr 24.1324, ips 47.73, lr 0.000195, time 75.8s
- Epoch 11: train_loss 0.0305, train_psnr 27.5193, val_loss 0.0390, val_psnr 24.1393, ips 45.71, lr 0.000193, time 77.1s
- Epoch 12: train_loss 0.0309, train_psnr 27.3144, val_loss 0.0388, val_psnr 24.1863, ips 44.74, lr 0.000190, time 78.8s
- Epoch 13: train_loss 0.0300, train_psnr 27.5652, val_loss 0.0388, val_psnr 24.1907, ips 46.03, lr 0.000188, time 77.5s
- Epoch 14: train_loss 0.0301, train_psnr 27.5801, val_loss 0.0386, val_psnr 24.2189, ips 47.94, lr 0.000185, time 75.3s
- Epoch 15: train_loss 0.0296, train_psnr 27.6557, val_loss 0.0384, val_psnr 24.2547, ips 48.54, lr 0.000182, time 74.8s
- Epoch 16: train_loss 0.0292, train_psnr 27.8281, val_loss 0.0382, val_psnr 24.2729, ips 46.38, lr 0.000179, time 77.6s
- Epoch 17: train_loss 0.0294, train_psnr 27.7415, val_loss 0.0381, val_psnr 24.2839, ips 49.22, lr 0.000176, time 74.5s
- Epoch 18: train_loss 0.0297, train_psnr 27.6480, val_loss 0.0381, val_psnr 24.2944, ips 48.81, lr 0.000172, time 75.0s
- Epoch 19: train_loss 0.0289, train_psnr 27.8341, val_loss 0.0380, val_psnr 24.3116, ips 47.94, lr 0.000168, time 75.8s
- Epoch 20: train_loss 0.0290, train_psnr 27.8579, val_loss 0.0377, val_psnr 24.3548, ips 48.59, lr 0.000164, time 75.2s
- Epoch 21: train_loss 0.0293, train_psnr 27.6748, val_loss 0.0378, val_psnr 24.3295, ips 48.36, lr 0.000159, time 75.1s
- Epoch 22: train_loss 0.0290, train_psnr 27.9133, val_loss 0.0377, val_psnr 24.3729, ips 48.61, lr 0.000155, time 75.7s
- Epoch 23: train_loss 0.0285, train_psnr 27.9138, val_loss 0.0375, val_psnr 24.3990, ips 48.83, lr 0.000150, time 74.7s
- Epoch 24: train_loss 0.0284, train_psnr 27.9366, val_loss 0.0375, val_psnr 24.4321, ips 46.70, lr 0.000145, time 76.1s
- Epoch 25: train_loss 0.0286, train_psnr 27.9673, val_loss 0.0374, val_psnr 24.4198, ips 46.33, lr 0.000140, time 77.2s
- Epoch 26: train_loss 0.0283, train_psnr 27.9832, val_loss 0.0374, val_psnr 24.3951, ips 47.91, lr 0.000135, time 75.8s
- Epoch 27: train_loss 0.0282, train_psnr 28.0707, val_loss 0.0372, val_psnr 24.4485, ips 48.74, lr 0.000130, time 75.1s
- Epoch 28: train_loss 0.0282, train_psnr 28.0753, val_loss 0.0371, val_psnr 24.4572, ips 38.03, lr 0.000125, time 88.7s
- Epoch 29: train_loss 0.0282, train_psnr 28.0163, val_loss 0.0368, val_psnr 24.5359, ips 49.19, lr 0.000120, time 77.6s
- Epoch 30: train_loss 0.0278, train_psnr 28.0670, val_loss 0.0368, val_psnr 24.4898, ips 48.61, lr 0.000114, time 77.8s
- Epoch 31: train_loss 0.0274, train_psnr 28.1765, val_loss 0.0366, val_psnr 24.5594, ips 42.83, lr 0.000109, time 79.7s
- Epoch 32: train_loss 0.0275, train_psnr 28.1187, val_loss 0.0366, val_psnr 24.5686, ips 48.67, lr 0.000103, time 74.4s
- Epoch 33: train_loss 0.0280, train_psnr 27.9967, val_loss 0.0365, val_psnr 24.5620, ips 48.86, lr 0.000098, time 74.3s
- Epoch 34: train_loss 0.0268, train_psnr 28.4013, val_loss 0.0365, val_psnr 24.5727, ips 48.36, lr 0.000092, time 74.9s
- Epoch 35: train_loss 0.0273, train_psnr 28.2358, val_loss 0.0362, val_psnr 24.6274, ips 48.61, lr 0.000087, time 74.9s
- Epoch 36: train_loss 0.0270, train_psnr 28.3548, val_loss 0.0362, val_psnr 24.6369, ips 48.50, lr 0.000081, time 74.6s
- Epoch 37: train_loss 0.0270, train_psnr 28.3628, val_loss 0.0361, val_psnr 24.6440, ips 48.73, lr 0.000076, time 75.6s
- Epoch 38: train_loss 0.0272, train_psnr 28.2768, val_loss 0.0359, val_psnr 24.6804, ips 47.88, lr 0.000071, time 75.7s
- Epoch 39: train_loss 0.0278, train_psnr 28.0191, val_loss 0.0362, val_psnr 24.6124, ips 48.94, lr 0.000066, time 75.8s
- Epoch 40: train_loss 0.0269, train_psnr 28.3041, val_loss 0.0359, val_psnr 24.6808, ips 49.11, lr 0.000061, time 75.7s
- Epoch 41: train_loss 0.0272, train_psnr 28.2628, val_loss 0.0358, val_psnr 24.7112, ips 48.95, lr 0.000056, time 76.0s
- Epoch 42: train_loss 0.0268, train_psnr 28.3445, val_loss 0.0359, val_psnr 24.6922, ips 48.70, lr 0.000051, time 77.5s
- Epoch 43: train_loss 0.0265, train_psnr 28.4925, val_loss 0.0358, val_psnr 24.7200, ips 45.29, lr 0.000046, time 78.1s
- Epoch 44: train_loss 0.0267, train_psnr 28.3939, val_loss 0.0357, val_psnr 24.7343, ips 48.97, lr 0.000042, time 74.4s
- Epoch 45: train_loss 0.0271, train_psnr 28.2487, val_loss 0.0359, val_psnr 24.7076, ips 48.99, lr 0.000037, time 75.4s
- Epoch 46: train_loss 0.0265, train_psnr 28.4425, val_loss 0.0357, val_psnr 24.7324, ips 47.28, lr 0.000033, time 75.9s
- Epoch 47: train_loss 0.0265, train_psnr 28.4505, val_loss 0.0355, val_psnr 24.7692, ips 48.65, lr 0.000029, time 75.6s
- Epoch 48: train_loss 0.0269, train_psnr 28.2506, val_loss 0.0357, val_psnr 24.7325, ips 45.23, lr 0.000025, time 77.3s
- Epoch 49: train_loss 0.0268, train_psnr 28.4682, val_loss 0.0356, val_psnr 24.7601, ips 48.85, lr 0.000022, time 75.4s
- Epoch 50: train_loss 0.0263, train_psnr 28.4565, val_loss 0.0355, val_psnr 24.7699, ips 48.69, lr 0.000019, time 74.9s
- Epoch 51: train_loss 0.0263, train_psnr 28.4622, val_loss 0.0354, val_psnr 24.7910, ips 47.94, lr 0.000016, time 75.0s
- Epoch 52: train_loss 0.0266, train_psnr 28.4611, val_loss 0.0356, val_psnr 24.7589, ips 48.94, lr 0.000013, time 75.2s
- Epoch 53: train_loss 0.0264, train_psnr 28.4684, val_loss 0.0355, val_psnr 24.7682, ips 47.01, lr 0.000011, time 75.8s

## Qualitative Observations

- The identity baseline is already strong on GoPro, so any single-image restoration model needs to clear a fairly high floor.
- `NAFNetLiteBaseline` improves over identity on both validation and test, which confirms that the paired pipeline and restoration model are working end to end.

## Issues Before Stronger Models

- This run uses a modest single-image restoration baseline, so model capacity and training duration are still conservative.
- Validation is derived from only three sequences, so conclusions should be treated as directional rather than final.
- The best checkpoint is selected by `val_psnr`, but worker and batch-size settings are still conservative rather than explicitly tuned.
- Full-resolution evaluation is deterministic, but training still samples one random patch per frame per epoch; this run plateaued before the 60-epoch target, so additional epochs alone are unlikely to change the result materially.
