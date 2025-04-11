# Agriculture Data

## Features

- Detection of multiple crop conditions:
  - Nutrient deficiency
  - Drydown
  - Planter skip
  - Water issues

## Dataset

Using Agriculture-Vision Dataset which includes:

- High-resolution aerial images
- Multiple spectral bands (RGB + NIR)
- Annotated masks for different crop conditions
- Two dataset options:
  - Full dataset (21GB)
  - Small dataset (4GB) consist of 18k samples

## Setup

### Installation

1. Clone the repository:

```bash
git clone https://github.com/ucef-b/agri-data.git
cd agri-data
```

2. Install required packages:

```bash
pip install gdown
```

### Data Preparation

1. Download the dataset:

```bash
# For small dataset (4GB)
gdown --id 1NLMpnMk_XPiVJddvqmsoCRjqTfIS88ra
```

2. Process the data:

```bash
python DataPrepare.py --source working_dir \
                    --output "./val" \
                    --selected nutrient_deficiency drydown planter_skip water \
                    --img-size 256 256 \
                    --max-samples 50 \
                    --test-split 0.4
```

## Usage

Load and process the dataset:

```python
from DataLoader import DatasetLoader

dataset = DatasetLoader(
    working_path="train",
    batch_size=32,
    export_type="NDVI",  # Options: "RGBN", "NDVI", "RGB"
    outputs_type="mask_only",  # Options: "mask_only", "class_only", "both"
    augmentation=False,
    shuffle=True
)
```

## Project Structure

```
agri-data/
├── agriculture_vision.ipynb   # Main notebook for data processing
├── DataLoader.py             # Dataset loading utilities
├── DataPrepare.py           # Data preparation script
└── README.md
```

## Reference

- [Agriculture-Vision Dataset](https://www.agriculture-vision.com/agriculture-vision-2021/dataset-2021)
