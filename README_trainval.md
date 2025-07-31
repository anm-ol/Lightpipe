# Train/Validation Dataset Split Feature

This document describes the train/validation dataset split functionality added to the Lightpipe video generation pipeline.

## Overview

The train/validation split feature allows you to generate datasets where:
- Objects, materials, and HDRIs are split into separate train and validation sets
- No asset overlap between train and validation splits (ensuring disjoint datasets)
- Videos are organized into separate `train/` and `val/` output directories
- Split ratio is configurable through the config file

## Configuration

Add the following section to your config YAML file:

```yaml
settings:
  # ... other settings ...
  
  # Train/Validation split configuration
  dataset_split:
    train_ratio: 0.8 # 80% for training, 20% for validation
    ensure_disjoint_assets: true # Ensure objects/HDRIs don't overlap between train/val
```

### Configuration Options

- `train_ratio`: Float between 0.0 and 1.0 specifying the proportion of assets to use for training
- `ensure_disjoint_assets`: Boolean flag to ensure no asset overlap between splits
  - `true`: Assets are split into disjoint train/val sets (recommended for proper evaluation)
  - `false`: Uses original behavior with no asset separation

## Output Structure

When using train/val split, the output directory structure will be:

```
output_base_path/
├── train/
│   ├── 0000_color.mp4
│   ├── 0000_depth.mp4
│   ├── 0001_color.mp4
│   ├── 0001_depth.mp4
│   └── ...
└── val/
    ├── 0000_color.mp4
    ├── 0000_depth.mp4
    ├── 0001_color.mp4
    ├── 0001_depth.mp4
    └── ...
```

## Usage Examples

### Example 1: 80/20 Train/Val Split
```yaml
settings:
  num_videos: 100
  dataset_split:
    train_ratio: 0.8
    ensure_disjoint_assets: true
```
Result: 80 training videos, 20 validation videos with no shared assets.

### Example 2: 70/30 Train/Val Split
```yaml
settings:
  num_videos: 50
  dataset_split:
    train_ratio: 0.7
    ensure_disjoint_assets: true
```
Result: 35 training videos, 15 validation videos with no shared assets.

### Example 3: Disable Train/Val Split
```yaml
settings:
  num_videos: 50
  dataset_split:
    ensure_disjoint_assets: false
```
Result: Original behavior - all 50 videos in single output directory.

## Testing

Use the provided test script to verify the split functionality:

```bash
python test_dataset_split.py configs/config_trainval.yaml
```

This will:
- Test asset splitting without running the full pipeline
- Show dataset split statistics
- Verify no asset overlap between train/val sets
- Report any errors in the configuration

## Implementation Details

### Asset Splitting Process

1. **Asset Discovery**: Scan directories for all available objects, materials, and HDRIs
2. **Shuffling**: Randomly shuffle each asset type using the provided seed
3. **Splitting**: Split each asset type according to `train_ratio`
4. **Sampling**: Sample assets for videos with replacement if needed

### Seed Handling

- Main seed from config is used for asset splitting (ensures reproducible splits)
- Train videos use main seed for sampling
- Validation videos use main seed + 1 for sampling (ensures different random sequences)

### Edge Cases

- If fewer assets available than videos requested, sampling uses replacement
- Empty asset directories will cause errors (intentional - indicates configuration issues)
- Non-existent asset paths will cause errors during asset discovery

## Configuration Files

- `configs/config_trainval.yaml`: Example configuration with train/val split enabled
- `configs/configv1.yaml`: Updated original config with train/val options

## Functions Added

### In `utils.py`:
- `get_all_data_paths()`: Get all available asset paths from directory
- `get_all_hdri_paths()`: Get all HDRI paths from directory  
- `split_assets_for_dataset()`: Split assets into train/val sets
- `sample_assets_for_videos()`: Sample assets for video generation
- `print_dataset_split_summary()`: Print split statistics

### In `main.py`:
- Updated `main_pipeline()` to handle train/val processing
- Added split detection and asset sampling logic
- Modified output directory structure for train/val splits
