# Interpretable Foundational LLM for Pathology

This project aims to build an interpretable system for classifying cervical cancer cells. It uses the Hindered Transformer (HiT), a vision transformer designed for interpretability, to classify cell images from the SIPaKMeD dataset. The long-term goal is to feed the model's outputs into a Large Language Model (LLM) to generate natural language explanations for its diagnostic decisions.

This repository is based on the official code for the paper: *Disentangling Visual Transformers: Patch-level Interpretability for Image Classification*.

## Project Setup

Follow these steps to set up the project environment and download the necessary assets.

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/luckyyducky/hit_cervical_project.git
cd hit_cervical_project
```

### 2. Set Up the Python Environment
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Install required packages
pip install torch torchvision timm numpy Pillow
```

### 3. Download Pretrained Models
The fine-tuning process requires a model pretrained on ImageNet. These model files are not tracked by Git and must be downloaded manually.

1.  **Download Files**: Go to the official [HiT Models Hugging Face Repository](https://huggingface.co/guillaumejs2402/HiT-models/tree/main). Download the following two files:
    * `hit_b-imagenet.pth`
    * `hit_b-imagenet.yaml`
2.  **Store Files**: Create a folder `hit_cervical_project/pretrained_models/` and place the two downloaded files inside it.

### 4. Prepare the Dataset
This project uses the SIPaKMeD dataset, which contains 4,049 annotated images of Pap smear cells across five categories.

1.  **Download Data**: The dataset is publicly available from the creators. Go to [http://www.cse.uoi.gr/~marina/sipakmed.html](https://www.cse.uoi.gr/~marina/sipakmed.html) and download the SIPaKMeD database.
2.  **Extract Data**: Unzip the downloaded files into a temporary location. You will have several folders named `im_(class)`, each containing a `CROPPED` subdirectory with the individual cell images.
3.  **Run Split Script**: This repository includes a script to automatically format the dataset. Make sure the paths in the `split_dataset.py` script are correct, then run it from the project's root directory:
    ```bash
    python split_dataset.py
    ```
    This will create a `dataset/` folder containing `train/`, `val/`, and `test/` subdirectories, populated with the images. This folder is ignored by Git.
