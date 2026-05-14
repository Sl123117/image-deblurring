# First Benchmark Report

## Train/Validation Split

- Seed: `42`
- Validation fraction from official training sequences: 10%
- Train sequences (19): GOPR0372_07_00, GOPR0372_07_01, GOPR0374_11_01, GOPR0374_11_02, GOPR0378_13_00, GOPR0379_11_00, GOPR0380_11_00, GOPR0384_11_01, GOPR0384_11_02, GOPR0384_11_03, GOPR0384_11_04, GOPR0385_11_00, GOPR0386_11_00, GOPR0477_11_00, GOPR0857_11_00, GOPR0868_11_01, GOPR0868_11_02, GOPR0871_11_01, GOPR0881_11_00
- Validation sequences (3): GOPR0374_11_00, GOPR0374_11_03, GOPR0884_11_00

## Model

- Architecture: `UNetDeblurModel`
- Trainable parameters: 837,827
- Total parameters: 837,827
- Base channels: 16
- Device used: `cpu`

## Training Setup

- Patch size: 256x256
- Batch size: 8
- Eval batch size: 2
- CPU threads: 2
- Epochs: 4
- Validate each epoch: False
- Checkpoint monitor: training loss
- Optimizer: Adam
- Learning rate: 0.0002
- Scheduler: StepLR(step_size=4, gamma=0.5)
- Loss: L1

## Metrics

- Identity validation: PSNR 24.0336, SSIM 0.7316
- Identity test: PSNR 25.6401, SSIM 0.7903
- U-Net validation: PSNR 24.0405, SSIM 0.7317
- U-Net test: PSNR 25.6549, SSIM 0.7906
- Validation improvement over identity: PSNR +0.0069, SSIM +0.0002
- Test improvement over identity: PSNR +0.0147, SSIM +0.0003

## Training History

- Epoch 1: train_loss 0.0302, train_psnr 27.3671, lr 0.000200, time 298.5s
- Epoch 2: train_loss 0.0302, train_psnr 27.5678, lr 0.000200, time 283.8s
- Epoch 3: train_loss 0.0311, train_psnr 27.3944, lr 0.000200, time 284.4s
- Epoch 4: train_loss 0.0305, train_psnr 27.5075, lr 0.000200, time 274.9s

## Qualitative Observations

- The identity baseline is already strong on GoPro, so the first U-Net benchmark needs to clear a fairly high floor.
- This run only improves very slightly over identity, so the network is learning some residual correction but not enough to materially change the benchmark yet.

## Issues Before Stronger Models

- This run uses a modest first-pass U-Net, so model capacity and training duration are still conservative.
- Validation is derived from only three sequences, so conclusions should be treated as directional rather than final.
- The checkpoint was selected by training loss because per-epoch full-resolution validation was disabled to keep the first benchmark tractable on CPU.
- The current setup is too weak to produce a meaningful margin over the identity baseline; before stronger model families, the next fix should be more training budget and validation-based checkpoint selection.
