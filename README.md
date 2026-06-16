# GPT-2 From Scratch

A minimal implementation of a GPT-style language model built completely from scratch using PyTorch. This project demonstrates how decoder-only Transformers work internally, including tokenization, self-attention, training, and autoregressive text generation.

## Features

- GPT-style Transformer architecture
- Multi-Head Self-Attention
- Feed Forward Networks
- Residual Connections & Layer Normalization
- Token & Positional Embeddings
- Causal Masking
- Custom Dataset Pipeline
- Training Loop with Checkpoint Saving
- Text Generation Script
- GPT-2 Tokenizer using Tiktoken

---

## Project Structure

```text
gpt2-from-scratch/
│
├── data/
│   └── input.txt
│
├── tokenizer.py
├── dataset.py
├── model.py
├── train.py
├── generate.py
│
├── gpt.pt
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Model Architecture

```text
Input Text
    ↓
Tokenizer (tiktoken)
    ↓
Token IDs
    ↓
Token Embeddings
    +
Positional Embeddings
    ↓
Transformer Blocks
    ├── Multi-Head Attention
    ├── Feed Forward Network
    ├── LayerNorm
    └── Residual Connections
    ↓
Linear Head
    ↓
Vocabulary Logits
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/pritam1952/gpt2-from-scratch.git

cd gpt2-from-scratch
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

Place your training corpus inside:

```text
data/input.txt
```

Example datasets:

- Shakespeare
- TinyStories
- WikiText
- Any custom text corpus

---

## Training

Run:

```bash
python train.py
```

Example model configuration:

```python
GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 32,
    "emb_dim": 32,
    "n_heads": 2,
    "n_layers": 1,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

Training saves model weights as:

```text
gpt.pt
```

---

## Text Generation

Generate text using a trained model:

```bash
python generate.py
```

Example:

```text
Prompt:
To be, or not to be

Generated:
To be, or not to be that is the question...
```

---

## Learning Objectives

This project was built to gain a deeper understanding of:

- Transformer Architecture
- Self-Attention Mechanism
- Language Modeling
- GPT Training Pipeline
- Tokenization
- Autoregressive Text Generation
- PyTorch Model Development

---

## Future Improvements

- GPT-2 124M Architecture
- Larger Context Length
- Mixed Precision Training
- GPU Optimization
- Model Evaluation Metrics
- Fine-Tuning Support
- Flash Attention
- Checkpoint Resuming

---

## Tech Stack

- Python
- PyTorch
- Tiktoken

---

## Results

Example training output:

```text
Epoch 1 | Loss: 4.23
Epoch 2 | Loss: 3.78
Epoch 3 | Loss: 3.41
```

Example generation:

```text
Prompt: ROMEO:
Generated:
ROMEO: My lord, I shall return before the morning light...
```

---

## License

This project is licensed under the MIT License.

---

## Author

**Pritam Kumar**

GitHub: https://github.com/pritam1952

If you found this project useful, consider giving it a ⭐ on GitHub.
