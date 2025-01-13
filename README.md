# High-Confidence Reconstruction for Laplace NMR Based on Uncertainty-Informed Deep Learning

# Overview
Laplace NMR presents a powerful detection technique for providing detailed insights into molecular dynamics and spin interactions by measuring relaxation and diffusion parameters, offering complementary chemical resolution to Fourier NMR. Spectrum reconstruction with accurate diffusion coefficients or relaxation time is essential for the Laplace NMR performance, but existing processing methods generally yield varying results due to the ill-posed nature of inverse Laplace Transform, making it difficult for the user to discern which parts of the estimation are accurate or which method is more reliable due to the lack of ideal references in practical applications.

This repository presents a deep learning-based approach that:
- **Accurately recovers** parameter distributions from exponential signals.
- **Provides uncertainty estimates** for each reconstruction, helping you assess confidence at every point in the spectrum.

By integrating uncertainty into the analysis, our method enhances the reliability of Laplace NMR interpretations and broadens its application in chemistry and materials science.

# Table of Contents
- [Requirements](#requirements)  
- [Usage](#usage)  
  - [Training](#training)  
    - [Default Training (Type: 'DOSY')](#default-training-type-dosy)  
    - [Custom Training (Type: 'T1T2')](#custom-training-type-t1t2)  
  - [Testing](#testing)  
- [Pre-trained Model Weights](#pre-trained-model-weights)
  - [Download Link](#download-link)
  - [Usage Guide](#usage-guide)
- [Visualization](#visualization)  
- [Contact](#contact)

# Requirements
To manage the environment, make sure you have [Conda](https://docs.conda.io/en/latest/) installed.

```bash
sh script/env.sh
conda activate DOSY
```
# Usage
## Training
By default, the training script is configured for diffusion experiments (Type: 'DOSY').
To switch to relaxation experiments (Type: 'T1T2'), simply pass it as a parameter.
## Default Training (Type: 'DOSY')
```shell
sh script/train.sh
```
## Custom Training (Type: 'T1T2')
```shell
sh script/train.sh T1T2
```
### Notes:
- The train.sh script accepts an optional parameter to set the Type. If no parameter is provided, it defaults to 'DOSY'.
- The Type parameter adjusts various configuration settings in config.py to suit different model architectures and dataset configurations.

## Testing
Use the provided Jupyter Notebook to evaluate the performance and reliability of the trained model:
- Testing Script: test.ipynb

# Pre-trained Model
## Download Link
We have provided pre-trained model weights to help you get started quickly. You can download the weights from our Google Drive folder using the link below:

[Download Pre-trained Weights](https://drive.google.com/drive/folders/1VEujZzFJZK7Nz4e7IaoWOnyyWYzn-vYO?usp=sharing)

## Usage Guide
### 1. Download the Weights
Click the link above to access and download the pre-trained weight files to your local machine.

### 2. Place the Weights
Move the downloaded weight files to the designated directory in the project, specifically the Result/ directory.

# Visualization
For further analysis and visual confirmation, MATLAB scripts are provided in:
- MATLAB/Results_Graph_CMP.mlx : Visualizes the results of the comparative.
- MATLAB/Results_Graph_SPL.mlx : Visualizes the results of the additional experiments.

These MATLAB files generate result plots of the data processed by the trained model.

## Contact
For questions, please reach out via email((33320231150340@stu.xmu.edu.cn)) or open a GitHub Issue. Pull requests are always welcome!

