# Plot Usage

This script allows you to visualize training curves (loss/accuracy) from saved
history JSON files, and compare multiple models.

## List

List every history JSON file inside the `checkpoints/` directory:

```bash
python plot.py --list
```
Arguments:

- None

This will display:
- All available history files with their full paths.

## Single Model Plot

Plot the training/validation curves for one history file:

```bash
python plot.py --single checkpoints/ModelA_history.json
```
Labels are optional:

```bash
python plot.py --single checkpoints/ModelA_history.json --labels "Model A"
```
Arguments:

- `--single <path>`\
Path to a single history JSON file.

- `--labels <label>`\
A single label to use as the plot title.

This will display both:
- Train vs Val Loss
- Train vs Val Accuracy

## Multiple Model Comparison

Plot validation accuracy curves for multiple models:
``` bash
python plot.py --multiple checkpoints/ModelA_history.json checkpoints/ModelB_history.json
```
Labels are optional:
``` bash
python plot.py \
    --multiple checkpoints/ModelA_history.json checkpoints/ModelB_history.json \
    --labels "Model A" "Model B"
```
Arguments:

- `--multiple <path1> <path2> ...`\
List of history JSON file paths to compare.

- `--labels <label1> <label2> ...`\
List of labels for each curve (must match number of history files).

This will show a single figure comparing validation accuracy across models.

## Compare Latest of All Models

Automatically detect all model types based on filenames and plot only the most recent run of each:

```bash
python plot.py --compare-latest
```
Examples of model types:
- `BaselineCNN_ModelA`
- `BaselineCNN_ModelB`
- `BatchnormCNN_ModelC`

This will:
- Identify all model prefixes,
- Find the newest timestamp for each.
- Plot their validation accuracy curves in one graph.

No manual file listing required.

## Compare the Latest Run of a Specific Model

Show the newest run for a specific model type:

```bash
python plot.py --latest <model name>
```
This will:
- Locate all history files for that model prefix
- Select the newest timestamp
- Plot full training curves as in the single-model plot

## List Available Model Names

If no model name is provided, `--latest` will list all available model types:

```bash
python plot.py --latest
```

Example output
```bash
Available model names:
  - BaselineCNN_ModelA
  - BaselineCNN_ModelB
  - BatchnormCNN_ModelC
  - DeepCNN_ModelD
```
