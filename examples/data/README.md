# Example Datasets

This directory contains datasets used in examples and tutorials.

## froggy.csv

**Source**: BANA Statistics Book

**Description**: Frog jump distance measurements and related variables from a real study.

**Variables**:
- `distance`: Jump distance in cm (main variable of interest)
- `duration`: Jump duration
- `jump_n`: Jump number (1, 2, or 3)
- `frog_type`: Type of frog ("pro" or other)
- `day`: Day of measurement
- `angle_*`, `velocity_*`: Additional measurements

**Usage Notes**:
- Some jumps have `distance = 0` (failed jumps) - filter these out for analysis
- Some measurements have `NA` values
- This dataset is used throughout the BANA statistics book examples
- Perfect for demonstrating descriptive statistics, hypothesis testing, and regression

**Example Usage**:
```python
import csv
from pathlib import Path
from real_simple_stats import descriptive_statistics as desc

# Load data
data_file = Path("examples/data/froggy.csv")
jump_distances = []
with open(data_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        distance = float(row['distance'])
        if distance > 0:  # Filter successful jumps
            jump_distances.append(distance)

# Calculate statistics
mean = desc.mean(jump_distances)
print(f"Mean jump distance: {mean:.2f} cm")
```

See these complete analysis examples:
- `examples/recipes/frog_jump_analysis.py` - Overall descriptive statistics
- `examples/recipes/frog_pro_vs_other.py` - Professional vs other frogs comparison

