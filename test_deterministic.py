#!/usr/bin/env python3
import blenderproc as bproc
"""
Test script to validate deterministic behavior of the pipeline.
This script simulates multiple task runs to ensure the same seed produces the same results.
"""
import os
import sys
import yaml
import random
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_deterministic_seeding():
    """Test that the seeding functions produce consistent results."""
    
    # Import functions from main
    from main import set_deterministic_seed, get_task_videos
    from utils import split_assets_for_dataset, sample_assets_for_videos
    
    master_seed = 42
    total_videos = 80
    
    print("Testing deterministic seeding...")
    
    # Test 1: Same seed should produce same task video distribution
    print("\n1. Testing task video distribution:")
    for num_tasks in [4, 8]:
        print(f"  With {num_tasks} tasks:")
        for task_id in range(num_tasks):
            start, end = get_task_videos(total_videos, task_id, num_tasks)
            print(f"    Task {task_id}: videos {start}-{end-1} ({end-start} videos)")
    
    # Test 2: Same seed should produce same randomization
    print("\n2. Testing randomization consistency:")
    results = []
    for run in range(3):
        set_deterministic_seed(master_seed, 0, 0, "test")
        random_vals = [random.random() for _ in range(5)]
        numpy_vals = [np.random.random() for _ in range(5)]
        results.append((random_vals, numpy_vals))
    
    # Check all runs produce same results
    all_same = all(results[0][0] == results[i][0] and 
                   np.allclose(results[0][1], results[i][1]) 
                   for i in range(1, 3))
    
    print(f"  Same seed produces same randomization: {'✓' if all_same else '✗'}")
    if all_same:
        print(f"    First 3 random values: {results[0][0][:3]}")
        print(f"    First 3 numpy values: {results[0][1][:3]}")
    
    # Test 3: Different seeds should produce different results
    print("\n3. Testing different seeds produce different results:")
    set_deterministic_seed(master_seed, 0, 0, "test")
    vals1 = [random.random() for _ in range(5)]
    
    set_deterministic_seed(master_seed, 0, 1, "test")  # Different video_idx
    vals2 = [random.random() for _ in range(5)]
    
    different = vals1 != vals2
    print(f"  Different contexts produce different values: {'✓' if different else '✗'}")
    
    print("\nDeterministic seeding test completed!")

if __name__ == "__main__":
    test_deterministic_seeding()
