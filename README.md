# ✍️ Handwritten Text Recognition

A deep learning project that recognizes and transcribes handwritten text using Microsoft's **TrOCR** (Transformer-based OCR) model and an **ANN** trained on the MNIST dataset.

---

## 📌 Overview

This project explores two approaches to handwritten text recognition:

1. **TrOCR (Microsoft)** — A pretrained Vision Encoder-Decoder model that reads handwritten images and outputs transcribed text.
2. **ANN on MNIST** — A custom Artificial Neural Network trained from scratch to classify handwritten digits (0–9).

Additionally, a comparative analysis of four model architectures (ANN, CNN, RNN, Hybrid) is visualized using bar charts and pie charts.

---

## 🧠 Models Used

| Model | Purpose |
|---|---|
| `microsoft/trocr-small-handwritten` | Handwritten word/sentence recognition |
| ANN (TensorFlow/Keras) | Handwritten digit classification (MNIST) |

---

## 📂 Project Structure

```
handwritten_text_generation_rithikaa.ipynb   # Main notebook
```

---

## 🔧 Requirements

Install the required libraries:

```bash
pip install numpy pandas seaborn matplotlib requests pillow transformers tensorflow scikit-learn
```

---

## 🗂️ Dataset

- **IAM Handwriting Database** — used for TrOCR evaluation
- **MNIST** — used for ANN digit classification
- Custom dataset (Google Drive) — 500 handwriting images for batch evaluation

---

## 👩‍💻 Author

**Rithikaa**  
Feel free to ⭐ this repo if you found it helpful!


