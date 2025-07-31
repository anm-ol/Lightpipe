#!/usr/bin/env python3
"""
Test script for the train/validation dataset split functionality.
This script tests the asset splitting without running the full rendering pipeline.
"""

import yaml
import sys
import os

# Add local path for imports
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'src'))

from utils import split_assets_for_dataset, sample_assets_for_videos, print_dataset_split_summary

def test_dataset_split(config_path):
    """Test the dataset split functionality."""
    print("Testing train/validation dataset split...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get configuration
    master_seed = config['project']['seed']
    dataset_config = config['settings'].get('dataset_split', {})
    train_ratio = dataset_config.get('train_ratio', 0.8)
    num_videos = config['settings']['num_videos']
    
    # Test asset splitting
    try:
        asset_split = split_assets_for_dataset(
            config['assets']['objects_dir'],
            config['assets']['materials_dir'], 
            config['assets']['hdri_dir'],
            train_ratio,
            master_seed
        )
        
        # Calculate video splits
        num_train_videos = int(num_videos * train_ratio)
        num_val_videos = num_videos - num_train_videos
        
        # Print summary
        print_dataset_split_summary(asset_split, num_train_videos, num_val_videos)
        
        # Test asset sampling
        print("Testing asset sampling...")
        train_assets = sample_assets_for_videos(asset_split, 'train', num_train_videos, master_seed)
        val_assets = sample_assets_for_videos(asset_split, 'val', num_val_videos, master_seed + 1)
        
        print(f"✓ Successfully sampled {len(train_assets['objects'])} train assets")
        print(f"✓ Successfully sampled {len(val_assets['objects'])} validation assets")
        
        # Verify no overlap in objects/materials/hdris between train and val
        train_obj_set = set(asset_split['train']['objects'])
        val_obj_set = set(asset_split['val']['objects'])
        train_mat_set = set(asset_split['train']['materials'])
        val_mat_set = set(asset_split['val']['materials'])
        train_hdri_set = set(asset_split['train']['hdris'])
        val_hdri_set = set(asset_split['val']['hdris'])
        
        obj_overlap = train_obj_set.intersection(val_obj_set)
        mat_overlap = train_mat_set.intersection(val_mat_set)
        hdri_overlap = train_hdri_set.intersection(val_hdri_set)
        
        if not obj_overlap and not mat_overlap and not hdri_overlap:
            print("✓ No asset overlap between train and validation sets")
        else:
            print(f"⚠ Warning: Found overlaps - Objects: {len(obj_overlap)}, Materials: {len(mat_overlap)}, HDRIs: {len(hdri_overlap)}")
        
        print("\n✓ Dataset split test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error testing dataset split: {e}")
        return False
    
    return True

if __name__ == "__main__":
    config_path = "configs/config_trainval.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found")
        sys.exit(1)
    
    success = test_dataset_split(config_path)
    sys.exit(0 if success else 1)
