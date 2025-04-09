# Seeing like a Cephalopod: Colour Vision with a Monochrome Event Camera

Code and analysis for Seeing like a Cephalopod: Colour Vision with a Monochrome Event Camera



<table align="center">
  <tr>
    <td align="center" style="border:none;">
      <a href="https://samiarja.github.io/earthobservation/" target="_blank">
        <img src="./figures/octopus_7591652.ico" alt="Project Page" width="50" style="padding:10px; background-color: #f5f5f5; border-radius: 10px; box-shadow: 2px 2px 12px #aaa;">
      </a>
    </td>
    <td align="center" style="border:none;">
      <a href="https://arxiv.org/pdf/2304.14125.pdf" target="_blank">
        <img src="./figures/arxiv.jpeg" alt="Paper" width="50" style="padding:10px; background-color: #f5f5f5; border-radius: 10px; box-shadow: 2px 2px 12px #aaa;">
      </a>
    </td>
    <td align="center" style="border:none;">
      <a href="./figures/2023CVPRW_DICMaxNEO_poster.pdf" target="_blank">
        <img src="./figures/poster_img.png" alt="Poster" width="68" style="padding:10px; background-color: #f5f5f5; border-radius: 10px; box-shadow: 2px 2px 12px #aaa;">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" style="border:none;">Project Page</td>
    <td align="center" style="border:none;">Paper</td>
    <td align="center" style="border:none;">Poster</td>
  </tr>
</table>



```
conda activate octoeye
```

Step 0: Record data with OctoEye (.raw) 
Step 1: Convert .raw to .h5 and multiple .txt

```
conda deactivate && python convert.py --input_file path/to/raw/file/*.raw
```

Step 2: E2VID frames reconstruction using the .h5 file

```
cd E2Calib-LCECalib
python offline_reconstruction.py  --freq_hz 5 --upsample_rate 20 --h5file path/to/raw/file/*.h5 --height 720 --width 1280
```

Step 3: Merge all .txt files into a single .es file

```
python txttoes.py
```

Step 4: Automatically label events to colour using local variance sweep

```
python auto_pixeltocolour.py
```

Step 5: Generate coloured event count/accumulated images

```
python colour_segmentation_frames.py
```

Step 6: Overlay the RGB labels on the greylevel reconstructed frames

```
python RGB_frames_reconstruction.py
```

Step 7: colour-focus-stack final image

```
cd Focus_Stacking
python run.py ./RGB_reconstructed_frames/red_image.png ./RGB_reconstructed_frames/green_image.png ./RGB_reconstructed_frames/blue_image.png
```

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

