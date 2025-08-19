# Multi-Task SLURM Pipeline

This pipeline has been updated to support distributed processing using SLURM's multi-task functionality while ensuring deterministic dataset generation.

## Key Features

### 1. Deterministic Dataset Generation
- The master seed in the config controls ALL randomization
- Same seed will always produce the exact same dataset regardless of task count
- Each video is assigned deterministic assets and randomization parameters

### 2. Multi-Task Distribution
- Videos are evenly distributed across tasks
- Each task processes a disjoint subset of videos
- GPU assignment is handled automatically to avoid conflicts
- Output files use global video indices to prevent naming conflicts

### 3. Asset Disjoint Splits
- When `ensure_disjoint_assets: true`, train and validation sets use completely different assets
- Asset splits are consistent across all tasks (determined by master seed)

## Usage

### Running with 4 tasks:
```bash
sbatch job_multi.sh
```

### Running with 8 tasks:
```bash
sbatch job_multi_8.sh
```

### Configuration
The config file `configs/configv1.yaml` controls:
- `project.seed`: Master seed for deterministic generation
- `settings.num_videos`: Total videos to generate (distributed across tasks)
- `settings.dataset_split.ensure_disjoint_assets`: Whether to separate train/val assets

## Task Distribution Examples

For 80 videos with different task counts:

**4 tasks:**
- Task 0: videos 0-19 (20 videos)
- Task 1: videos 20-39 (20 videos)  
- Task 2: videos 40-59 (20 videos)
- Task 3: videos 60-79 (20 videos)

**8 tasks:**
- Task 0: videos 0-9 (10 videos)
- Task 1: videos 10-19 (10 videos)
- Task 2: videos 20-29 (10 videos)
- Task 3: videos 30-39 (10 videos)
- Task 4: videos 40-49 (10 videos)
- Task 5: videos 50-59 (10 videos)
- Task 6: videos 60-69 (10 videos)
- Task 7: videos 70-79 (10 videos)

## GPU Assignment
- 2 GPUs are allocated per job
- Tasks automatically distribute across available GPUs
- Task with local_rank 0 uses GPU 0, local_rank 1 uses GPU 1, etc.

## Output Structure
```
output/
├── train/
│   ├── 0000_color.mp4    # First train video
│   ├── 0000_depth.mp4
│   ├── 0001_color.mp4
│   └── ...
└── val/
    ├── 0064_color.mp4    # First val video (after 64 train videos)
    ├── 0064_depth.mp4
    └── ...
```

## Testing Determinism
Run the test script to verify deterministic behavior:
```bash
cd /mnt/venky/ankitd/anmol/new_vace_training/Lightpipe
python test_deterministic.py
```

## Important Notes

1. **Seed Consistency**: The same seed will always produce the same dataset, regardless of how many tasks are used to generate it.

2. **Asset Distribution**: When using disjoint assets, the train/val split is determined globally and consistently across all runs with the same seed.

3. **File Naming**: Output files use global video indices, so multiple tasks won't create conflicting filenames.

4. **Error Handling**: Each task operates independently, so if one task fails, others continue processing.

5. **Memory and GPU**: Resources are automatically distributed based on the number of tasks and available hardware.
