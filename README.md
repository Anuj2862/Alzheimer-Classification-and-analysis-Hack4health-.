# AI 4 Alzheimer's - Hack4Health

![Project Banner](https://img.shields.io/badge/AI-Healthcare-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)

## Overview

**AI 4 Alzheimer's** is a deep learning solution designed to assist in the early detection and severity grading of Alzheimer's Disease using MRI scans. This project was developed for the **Hack4Health** hackathon.

The system utilizes a **DenseNet121** architecture with **Focal Loss** to handle class imbalances, achieving high accuracy in distinguishing between healthy and demented patients, as well as grading the severity of the disease.

## Key Features

-   **Automated Preprocessing**: Intelligent cropping, CLAHE (Contrast Limited Adaptive Histogram Equalization), and resizing of MRI scans.
-   **Dual-Mode Classification**:
    -   **Binary Mode**: Detects if a patient is Healthy or Demented.
    -   **Severity Mode**: Grades dementia into Mild, Very Mild, and Moderate.
-   **Robust Training**: Implements Stratified Shuffle Split, Weighted Random Sampling, and Focal Loss to ensure robust performance on imbalanced datasets.
-   **Modular Design**: Clean, modular codebase structured for scalability.

## Tech Stack

-   **Deep Learning**: PyTorch, Torchvision
-   **Image Processing**: OpenCV, PIL
-   **Data Manipulation**: Pandas, NumPy
-   **Machine Learning Utilities**: Scikit-Learn

## Project Structure

```bash
├── Datasets/               # Data directory (parquet files)
├── models/                 # Saved model weights
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── config.py           # Configuration parameters
│   ├── dataset.py          # Dataset classes and preprocessing
│   ├── model.py            # Model architectures (DenseNet, ResNet)
│   ├── train.py            # Main training script
│   └── utils.py            # Helper functions
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Anuj2862/Alzheimer-Classification-and-analysis.git
    cd Alzheimer-Classification-and-analysis
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the model, run the `src/train.py` script. You can toggle between "binary" and "severity" modes by modifying the `main()` call in the script.

```bash
python -m src.train
```

### Preprocessing Logic

The preprocessing pipeline ensures consistency across all MRI scans:
1.  **Cropping**: Removes empty black space around the brain.
2.  **CLAHE**: Enhances local contrast to highlight brain structures.
3.  **Resizing**: Standardizes inputs to 160x160 pixels.

## Results

-   **Binary Classification**: 99.4% Accuracy (Validation)
-   **Severity Classification**: 97.6% Accuracy (Validation)
*(Results based on initial experiments with the Kaggle MRI Dataset)*

## Contributors

-   **Anuj** - *Lead Developer*
-   *Hack4Health Team*
