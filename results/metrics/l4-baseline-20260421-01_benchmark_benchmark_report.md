# First Benchmark Report

## Run

- Artifact prefix: `l4-baseline-20260421-01`

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
- U-Net validation: PSNR 24.3450, SSIM 0.7426
- U-Net test: PSNR 26.2236, SSIM 0.8058
- Validation improvement over identity: PSNR +0.3114, SSIM +0.0076
- Test improvement over identity: PSNR +0.5835, SSIM +0.0128

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

## Qualitative Observations

- The identity baseline is already strong on GoPro, so the first U-Net benchmark needs to clear a fairly high floor.
- The U-Net improves over identity on both validation and test, which confirms that the paired pipeline and restoration model are working end to end.

## Issues Before Stronger Models

- This run uses a small CPU-friendly U-Net, so capacity and training duration are both conservative.
- Validation is derived from only three sequences, so conclusions should be treated as directional rather than final.
- The best checkpoint is selected by `val_psnr`, but worker and batch-size calibration are intentionally deferred to the dedicated GPU execution pass.
- Full-resolution evaluation is deterministic, but training still samples one random patch per frame per epoch; longer runs should improve stability.
