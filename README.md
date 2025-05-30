# IF3270 Machine Learning - Tugas Besar 2


## 📘 Project Overview

This project focuses on the implementation and analysis of **Convolutional Neural Networks (CNN)**, **Simple Recurrent Neural Networks (Simple RNN)**, and **Long Short-Term Memory (LSTM)** networks. It aims to provide a deep understanding of neural network architectures by implementing **forward propagation from scratch** and comparing it with models built using **Keras**.

## 📚 Contents

* `src/`: Python source code and notebooks for model training, analysis, and forward propagation implementation.
* `doc/`: Project report in PDF format detailing implementation, experiments, results, and member contributions.

## 🔧 Setup and Execution

1. Clone the repository:

   ```bash
   git clone https://github.com/0x0wen/CNNxRNN
   cd CNNxRNN
   cd src
   ```

2. Install required dependencies
    ```bash
    uv sync
    ```

3. Go into the corresponding directory
    ```bash
    cd {model_dir}
    ```

4. Run the main program
    ```bash
    uv run main.py
    ```

## Running SimpleRNN

1. Training the Model
    ```bash
    uv run python main.py --train
    ```
2. Testing the From-Scratch Implementation
    ```bash
    uv run python main.py --test-scratch MODEL_NAME_OR_CONFIG
    ```
3. Testing the From-Scratch Implementation With Batch Size
    ```bash
    uv run python main.py --test-scratch MODEL_NAME_OR_CONFIG --inference-batch-size BATCH_SIZE
    ```
4. Testing the From-Scratch Implementation With Backward
    ```bash
    uv run python main.py --test-scratch MODEL_NAME_OR_CONFIG --demo-backward
    ```


## 🧠 Implemented Tasks

### ✅ CNN (CIFAR-10)

* Training with Keras (Conv2D, Pooling, Dense)
* Hyperparameter analysis (layer depth, filter size/count, pooling types)
* Manual forward propagation

### ✅ Simple RNN & LSTM (NusaX-Sentiment)

* Preprocessing (tokenization & embedding)
* Text classification using Simple RNN and LSTM
* Hyperparameter analysis (layers, cell count, bidirectional/unidirectional)
* Manual forward propagation

## 👥 Team Members & Responsibilities

| Name            | NIM         | Role                                             |
| --------------- | ----------- | ------------------------------------------------ |
| Owen Tobias Sinurat   | 13522131 | LSTM implementation & analysis                    |
| Ahmad Thoriq Saputra    | 13522141 | RNN implementation & analysis                         |
| Muhammad Fatihul Irhab | 13522143 | CNN implementation & analysis       |
