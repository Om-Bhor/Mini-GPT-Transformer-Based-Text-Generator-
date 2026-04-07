<h3>
  # 🧠 Mini GPT (Transformer-Based Text Generator)

## 📌 Overview

This project implements a **mini GPT (Generative Pre-trained Transformer)** from scratch using PyTorch.
The model is trained on a text dataset and learns to generate human-like text by predicting the next word in a sequence.

It demonstrates the **core concepts behind modern LLMs** such as GPT, including self-attention, multi-head attention, and transformer blocks.

---

## 🚀 Features

* Built from scratch using PyTorch (no high-level transformer libraries)
* Implements **self-attention mechanism**
* Multi-head attention for better context understanding
* Word-level text generation
* Streamlit-based web interface for interactive text generation

---

## 🏗️ Model Architecture

The model follows a **decoder-only Transformer architecture**, similar to GPT.

### 1. Embeddings

* **Token Embedding**: Converts words into dense vectors
* **Positional Embedding**: Adds positional information to maintain sequence order

---

### 2. Transformer Blocks

Each block consists of:

#### 🔹 Multi-Head Self Attention

* Learns relationships between words in a sequence
* Uses **Query, Key, Value (QKV)** mechanism
* Applies **causal masking** to prevent future token access

#### 🔹 Feed Forward Network

* Fully connected layers for feature transformation
* Adds non-linearity using ReLU activation

#### 🔹 Residual Connections + Layer Normalization

* Helps stabilize training
* Prevents vanishing gradient problem

---

### 3. Output Layer

* Linear layer maps embeddings to vocabulary size
* Softmax converts logits into probabilities

---

## 🔄 Training Process

1. Text is tokenized into word indices
2. Input sequences (`x`) and target sequences (`y`) are created
3. Model predicts next word for each position
4. Loss is computed using **Cross-Entropy Loss**
5. Backpropagation updates model weights using AdamW optimizer

---

## ✨ Text Generation

The model generates text using:

1. Start with a prompt
2. Predict next word probabilities
3. Sample next word using softmax distribution
4. Append to sequence
5. Repeat for desired length

---

## 🌐 Streamlit App

A simple web interface allows users to:

* Enter a prompt
* Generate text using the trained model
* View output instantly

Run the app:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
mini-gpt/
│── train.py        # Model training script
│── app.py          # Streamlit web app
│── model.pt        # Trained model weights
```

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Streamlit
* Hugging Face Datasets

---

## 📊 Limitations

* Small model size → limited text quality
* Word-level tokenization → less flexibility
* Trained on limited data

---

## 🚀 Future Improvements

* Use subword tokenization (e.g., GPT-2 tokenizer)
* Train on larger datasets
* Increase model size and training steps
* Add chat-style interface

---

## 🎯 Conclusion

This project provides a **hands-on understanding of how GPT models work internally**, making it a great starting point for learning about Large Language Models and Generative AI.

---

</h3>
