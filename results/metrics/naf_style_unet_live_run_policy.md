# `naf_style_unet` Live Run Policy

Use `l4-charb-cosine-20260512-01` as the locked baseline for matched-epoch validation PSNR comparisons.

## Baseline Reference

- epoch `12`: `24.2483`
- epoch `24`: `24.4079`
- epoch `36`: `24.5593`
- epoch `48`: `24.6174`

## Early-Stop Thresholds

Stop the live `naf_style_unet` run early if validation PSNR is below:

- epoch `12`: `24.17`
- epoch `24`: `24.31`
- epoch `36`: `24.44`
- epoch `48`: `24.55`

## Interpretation

- Small early lag is acceptable.
- If the model remains more than roughly `0.10-0.12 dB` behind through the mid or late run, do not spend the remaining GPU time.
- If the model clears these checkpoints and completes, require at least `+0.10 dB` test PSNR over `l4-charb-cosine-20260512-01` with no test SSIM regression before keeping it as the new custom baseline.
