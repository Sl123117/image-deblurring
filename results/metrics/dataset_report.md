# Dataset Inspection Report

- Dataset found: GOPRO_Large
- Dataset root: `/Users/srirang/Desktop/Projects/image-deblurring/data/raw/GOPRO_Large`
- Total verified pairs: 3214
- Total sequences: 33
- Image sizes: {'1280x720': 3214}
- Auxiliary per-sequence directories: {'blur_gamma': 33}

## Split Summary

- `test`: 11 sequences, 1111 verified blur/sharp pairs, sizes {'1280x720': 1111}
- `train`: 22 sequences, 2103 verified blur/sharp pairs, sizes {'1280x720': 2103}

## Pairing Logic

- Each sequence uses `blur/<filename>` paired with `sharp/<filename>`.
- Pair matching is one-to-one by identical filename within the same sequence directory.

## Validation Recommendation

- Create validation data by holding out whole training sequences with a fixed random seed.
- Suggested holdout fraction: 10% of training sequences, with seed `42`.

## Issues

- No missing pairs or resolution mismatches were found in the verified data.
