# Early Stop Note

- Run name: `l4-motion-routed-20260513-01`
- Model: `motion_routed_unet`
- Status: stopped manually on May 13, 2026 before benchmark completion

## Reason

This run was stopped because the validation curve was consistently behind the locked plain-U-Net baseline under the same training recipe and budget.

Matched checkpoints from the live log:

- Epoch `24`: motion-routed `24.318` val PSNR vs baseline `24.4079`
- Epoch `36`: motion-routed `24.400` val PSNR vs baseline `24.5593`
- Epoch `38`: motion-routed `24.433` val PSNR vs baseline `24.5555`

The gap remained material into the cosine phase, so the expected remaining gain was unlikely to recover the deficit by epoch `60`.

## Decision

- Record `motion_routed_unet v1` as a miss
- Do not spend more L4 time tuning this architecture in place
- Move to a different stronger architecture idea for the next benchmark

## Last Observed Live Metrics

- Epoch `38`
- Learning rate: `0.000071`
- Train loss: `0.0273`
- Train PSNR: `28.286`
- Validation loss: `0.0371`
- Validation PSNR: `24.433`
- Throughput: `64.49` images/sec
