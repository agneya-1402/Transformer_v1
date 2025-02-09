# Transformer from Scratch using PyTorch

This repository contains a Python implementation of the Transformer model built entirely from scratch using PyTorch. The implementation was inspired by the [DataCamp tutorial on Building a Transformer with PyTorch](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch) and significantly influenced by the original research paper [*Attention is All You Need*](https://arxiv.org/pdf/1706.03762v7).

## Overview

The Transformer model revolutionized Natural Language Processing (NLP) by introducing a self-attention mechanism that allows the model to weigh the influence of different tokens in the input sequence. This implementation covers the core components:
- **Multi-Head Attention:** Enables the model to focus on different parts of the input sequence simultaneously.
- **Position-wise Feed-Forward Networks:** Applies non-linear transformations to each token independently.
- **Positional Encoding:** Adds information about the position of tokens in the sequence.
- **Encoder and Decoder Layers:** Combines the above components with residual connections and layer normalization.

## Getting Started

### Prerequisites

- **Python 3.7+**
- **PyTorch** (tested with version 1.8+)

### Installation

Install the required dependencies using pip:

```bash
pip3 install torch torchvision torchaudio
```

Or, if you are using Conda:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

### Repository Contents
main_1.py: The main Python file containing the full Transformer implementation.
README.md: This file.

### How to Run
Since this implementation is provided as a standalone Python file, you can run the model by executing the script in your terminal:

```bash
python main_1.py
```
### When executed, the script will:
* Instantiate the Transformer model with example hyperparameters.
* Create dummy source and target sequences.
* Perform a forward pass through the model.
* Print the output shape (expected: batch_size x sequence_length x target_vocab_size).

### Code Structure
* MultiHeadAttention: Implements the scaled dot-product attention mechanism with multiple attention heads.
* PositionWiseFeedForward: A two-layer feed-forward network with ReLU activation.
* PositionalEncoding: Adds sinusoidal positional encodings to the input embeddings.
* EncoderLayer: Combines multi-head attention and feed-forward networks with residual connections and layer normalization.
* DecoderLayer: Similar to the encoder layer but includes an additional cross-attention mechanism.
* Transformer: Stacks multiple encoder and decoder layers to form the complete Transformer architecture.

## References
DataCamp Tutorial: Building a Transformer with PyTorch<br>
Original Research Paper: Attention is All You Need

## Acknowledgments
A special thanks to the authors of the original Transformer paper and the DataCamp team for their comprehensive tutorial, which provided invaluable guidance for this project.
