# High-Confidence Reconstruction for Laplace NMR Based on Uncertainty-Informed Deep Learning

## Overview
Laplace NMR is a powerful technique for probing molecular dynamics and spin interactions by measuring relaxation and diffusion parameters, complementing the chemical resolution of Fourier NMR. However, accurately reconstructing spectra with precise diffusion coefficients or relaxation times is challenging due to the ill-posed nature of the inverse Laplace Transform. Existing methods often produce inconsistent results, making it difficult to assess the reliability of the estimations.

To address these challenges, we developed a deep learning-based approach that:
- Accurately recovers parameter distributions from exponential signals.
- Provides uncertainty estimates for each reconstruction, enabling confidence assessment across different spectral regions.

This uncertainty-informed framework enhances the reliability of Laplace NMR interpretations, offering a clearer and more dependable analysis. Our method facilitates broader applications in fields such as chemistry and materials science by providing a more robust and trustworthy tool for researchers.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
    - [Default Training (Type: 'DOSY')](#default-training-type-dosy)
    - [Custom Training (Type: 'T1T2')](#custom-training-type-t1t2)
  - [Evaluation](#evaluation)
  - [Testing](#testing)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contact](#contact)
- [License](#license)

## Requirements

Ensure you have [Conda](https://docs.conda.io/en/latest/) installed for environment management.

```shell
sh script/env.sh
conda activate DOSY
```

## Usage
### Training
Run the training script. By default, the Type is set to 'DOSY' for diffusion experiments. To use 'T1T2' for relaxation experiments, pass it as an argument.
#### Default Training (Type: 'DOSY')
```shell
sh script/train.sh
```
#### Custom Training (Type: 'T1T2')
```shell
sh script/train.sh T1T2
```
Notes:
- The train.sh script accepts an optional parameter to set the Type. If no parameter is provided, it defaults to 'DOSY'.
- The Type parameter adjusts various configuration settings in config.py to suit different model architectures and dataset configurations.

### Testing
Use the provided Jupyter Notebook to test the trained model's performance and reliability:
- Testing Script: test.ipynb

## Contact
If you have any questions, please contact us via email at 1015149201@qq.com or through GitHub Issues. Pull requests are highly welcomed!

