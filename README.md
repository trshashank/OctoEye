# OctoEye: Event-based Colour Perception Inspired by Cephalopods Vision

Step 0: Record some event data

Step 1: denoise events and convert .raw to .h5 and .txt
conda deactivate && python3 convert.py --input_file /media/samiarja/USB/raw/recording_2024-12-07_16-13-36_pattern9_backup.raw

Step 2: Convert all .txt to .es
python txttoes.py

Step 3: Create binary mask
python octoeye_mask_overlay.py

Step 4: Label events to RGB
python pixeltocolour.py

Step 5: Generate colour frames
python colour_segmentation_frames.py

Step 6: E2VID reconstruction and RGB colour overlay
Use E2Calib code

Step 7: Overlay colours on the frame reconstruction

Step 8: Make figures for paper

# Summary



# Setup

## Requirements

- python: 3.9.x, 3.10.x

## Tested environments
- Ubuntu 22.04
- Conda 23.1.0
- Python 3.9.18

## Installation

```sh
git clone https://github.com/samiarja/OctoEye.git
cd OctoEye
conda env create -f environment.yml
conda activate octoeye
python3 -m pip install -e .
```

