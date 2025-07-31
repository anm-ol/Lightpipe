# Train/Validation Dataset Split - Implementation Summary

## Files Modified/Created

### 1. `configs/configv1.yaml` - Updated
- Added `dataset_split` section with `train_ratio` and `ensure_disjoint_assets` options

### 2. `configs/config_trainval.yaml` - New
- Complete example configuration demonstrating train/val split feature
- Uses 80/20 split ratio with disjoint assets enabled

### 3. `src/utils.py` - Updated
Added new functions:
- `get_all_data_paths()` - Get all available data paths from directory
- `get_all_hdri_paths()` - Get all HDRI file paths from directory  
- `split_assets_for_dataset()` - Split assets into train/val sets ensuring no overlap
- `sample_assets_for_videos()` - Sample assets for video generation with replacement
- `print_dataset_split_summary()` - Print detailed split statistics

### 4. `src/main.py` - Updated
Major changes to `main_pipeline()`:
- Added train/val split configuration parsing
- Implemented asset splitting logic with disjoint sets
- Modified video generation loop to process train and val splits separately
- Updated output directory structure to create `train/` and `val/` subdirectories
- Enhanced HDRI selection to respect asset splits
- Added comprehensive logging for split processing

### 5. `test_dataset_split.py` - New
- Standalone test script to validate train/val split functionality
- Tests asset splitting, sampling, and overlap detection
- Provides detailed reporting without running full pipeline

### 6. `README_trainval.md` - New
- Comprehensive documentation for the train/val split feature
- Configuration examples and usage instructions
- Implementation details and edge case handling

## Key Features Implemented

### ✅ Disjoint Asset Sets
- Objects, materials, and HDRIs are split into separate train/val sets
- No overlap between train and validation assets (when `ensure_disjoint_assets: true`)
- Reproducible splits using configurable seed

### ✅ Configurable Split Ratios
- `train_ratio` parameter controls the proportion of assets for training
- Automatic calculation of train/val video counts
- Support for any ratio between 0.0 and 1.0

### ✅ Organized Output Structure
```
output_base_path/
├── train/
│   ├── 0000_color.mp4
│   ├── 0000_depth.mp4
│   └── ...
└── val/
    ├── 0000_color.mp4
    ├── 0000_depth.mp4
    └── ...
```

### ✅ Backward Compatibility
- Original behavior preserved when `ensure_disjoint_assets: false`
- Existing config files work without modification (uses defaults)

### ✅ Comprehensive Logging
- Detailed split statistics and progress reporting
- Asset count summaries for train/val sets
- Clear indication of which split is being processed

### ✅ Error Handling
- Graceful handling of insufficient assets (uses sampling with replacement)
- Validation of asset directories and paths
- Clear error messages for configuration issues

## Usage Examples

### Basic Train/Val Split (80/20)
```bash
python src/main.py --config configs/config_trainval.yaml
```

### Test Split Configuration
```bash
python test_dataset_split.py configs/config_trainval.yaml
```

### Custom Configuration
```yaml
settings:
  num_videos: 100
  dataset_split:
    train_ratio: 0.7  # 70% train, 30% val
    ensure_disjoint_assets: true
```

## Benefits

1. **Proper Evaluation**: Disjoint train/val sets ensure unbiased model evaluation
2. **Reproducible**: Seed-based splitting ensures consistent results across runs
3. **Flexible**: Configurable split ratios for different experimental needs
4. **Organized**: Clear output structure for easy dataset management
5. **Scalable**: Handles large asset collections efficiently
6. **Compatible**: Works with existing BlenderProc pipeline without breaking changes

## Next Steps

1. Run `test_dataset_split.py` to validate the implementation
2. Adjust config parameters as needed for your dataset requirements
3. Run the full pipeline with `python src/main.py --config configs/config_trainval.yaml`
4. Verify output directory structure and video generation
