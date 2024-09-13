# AudioBERT Implementation

This repository contains a comprehensive implementation of the **AudioBERT** system as described in the research paper:

> **AudioBERT: Audio Knowledge Augmented Language Model**

The system integrates auditory knowledge into a language model to enhance its understanding and processing of audio-related information.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Dependencies](#dependencies)
- [Setup Instructions](#setup-instructions)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Visualization and Reporting](#visualization-and-reporting)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Introduction

This implementation covers all major components of the AudioBERT system, including:

1. **AuditoryBench Dataset Generation**: Processes and generates datasets for training and evaluation, including auditory prompts and labels.
2. **Auditory Knowledge Span Detector**: A BERT-based model that detects spans in text that refer to auditory information.
3. **CLAP Integration**: Incorporates Contrastive Language-Audio Pretraining to align audio and text representations.
4. **AudioBERT Core**: The main model that integrates audio embeddings into the language model using LoRA (Low-Rank Adaptation).
5. **Evaluation and Baselines**: Tools to evaluate the model's performance against baseline models.
6. **Visualization and Reporting**: Functions to visualize training progress, attention mechanisms, and model predictions.

---

## Features

- **Full Implementation of AudioBERT**: A detailed and advanced implementation covering all aspects of the system.
- **Modular Code Structure**: Organized code for easy customization and extension.
- **Data Augmentation and Preprocessing**: Comprehensive data augmentation techniques using NLTK's WordNet.
- **Integrated Pipelines**: Training, evaluation, and prediction pipelines are seamlessly integrated.
- **Visualization Tools**: Includes tools for analyzing and visualizing model performance.
- **Command-Line Interface**: Easy interaction through a well-defined CLI.
- **Advanced Techniques**: Incorporates LoRA, mixed-precision training, and gradient accumulation for efficiency.

---

## Architecture Overview

The AudioBERT system consists of multiple interconnected components:

- **AuditoryBench Dataset**: A curated dataset with prompts and labels for auditory tasks.
- **Auditory Knowledge Span Detector**: Identifies and labels spans in text that relate to auditory information.
- **CLAP Model**: Learns joint embeddings for audio and text using contrastive learning.
- **AudioBERT Model**: Enhances a BERT-based language model with audio embeddings and LoRA.
- **Baselines**: Comparison with other models like BERT, RoBERTa, etc.

---

## Dependencies

Ensure the following dependencies are installed:

- **Programming Language**: Python 3.8 or higher
- **Core Libraries**:
  - PyTorch
  - Transformers
  - Torchaudio
  - Librosa
  - FAISS
  - NumPy
  - Pandas
  - Matplotlib
  - Scikit-learn
  - TensorBoard
  - PyYAML
  - tqdm
  - scikit-image
  - seaborn
  - NLTK
  - sentencepiece

Install them using:

```bash
pip install -r requirements.txt
```

*Note: A `requirements.txt` file is provided with all the dependencies listed.*

---

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/sanowl/AudioBERT.git
   cd audiobert
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**

   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

5. **Prepare Configuration File**

   Edit the `config.yaml` file to set up paths and hyperparameters.

   ```yaml
   dataset:
     csv_path: 'path/to/laion_audio.csv'
   training:
     epochs: 5
     batch_size: 16
     learning_rate: 1e-5
     weight_decay: 0.01
     max_grad_norm: 1.0
     early_stopping_patience: 3
   clap_training:
     epochs: 10
     batch_size: 8
     learning_rate: 1e-4
   audiobert_training:
     epochs: 20
     batch_size: 4
     learning_rate: 3e-4
     max_grad_norm: 1.0
   ```

---

## Configuration

- **config.yaml**: Central configuration file where dataset paths, training parameters, and model settings can be adjusted.

---

## Data Preparation

1. **LAION-Audio-630K Dataset**

   - Download the dataset and place it in the appropriate directory.
   - Update the `csv_path` in `config.yaml` with the path to the dataset CSV file.

2. **Data Augmentation**

   - The script includes data augmentation techniques using NLTK's WordNet for synonym replacement.
   - Ensure NLTK data is downloaded as per the setup instructions.

---

## Training

Run the training script with:

```bash
python audiobert.py --train
```

This will:

- Load and preprocess the dataset.
- Train the **Auditory Knowledge Span Detector**.
- Train the **CLAP model** for audio-text embedding alignment.
- Train the **AudioBERT model** integrating audio embeddings.

**Note**: Training may require a significant amount of computational resources. It is recommended to use a machine with a dedicated GPU.

---

## Evaluation

To evaluate the trained models:

```bash
python audiobert.py --evaluate
```

This will:

- Load the saved models.
- Evaluate on the test dataset.
- Provide metrics like accuracy and F1-score.

---

## Prediction

To make predictions on new input text:

```bash
python audiobert.py --predict "Your input text here."
```

The script will output the model's prediction and any relevant information.

---

## Visualization and Reporting

- **TensorBoard**: Training logs are saved for visualization. Launch TensorBoard with:

  ```bash
  tensorboard --logdir=runs
  ```

- **Attention Visualization**: The script includes functions to visualize attention maps for given inputs.

- **Learning Curves**: Plots of training and validation losses are saved as images in the `outputs/` directory.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- **Original Paper**: [AudioBERT: Audio Knowledge Augmented Language Model](https://arxiv.org/abs/2409.08199)
- **Libraries and Frameworks**:
  - [PyTorch](https://pytorch.org/)
  - [Transformers](https://huggingface.co/transformers/)
  - [Torchaudio](https://pytorch.org/audio/)
  - [Librosa](https://librosa.org/)
  - [FAISS](https://github.com/facebookresearch/faiss)
  - [NLTK](https://www.nltk.org/)

---

## Contact

For any questions or issues, please open an issue on the GitHub repository or contact **san.hashimhama@outlook.com**.

---

**Important Notes**

- **Data and Model Paths**: Ensure that all paths in the configuration file are correctly set to your local environment.

- **Model Checkpoints**: The training scripts save model checkpoints after each epoch. Ensure sufficient storage space is available.

- **Hardware Requirements**: Training the models, especially the larger ones, can be resource-intensive. Adjust batch sizes and other parameters if memory issues are encountered.

- **Troubleshooting**:
  - **Memory Errors**: Reduce batch sizes or use gradient accumulation if memory limitations are encountered.
  - **Module Not Found**: Ensure all dependencies are installed in the active Python environment.

- **Extending the Project**:
  - Additional datasets can be added, or other models can be integrated.
  - Experimenting with hyperparameters may improve model performance.

---

This implementation and guide aim to assist in exploring the capabilities of the AudioBERT system effectively.

---

*Note*: Replace `path/to/laion_audio.csv` with the actual path to your dataset.
