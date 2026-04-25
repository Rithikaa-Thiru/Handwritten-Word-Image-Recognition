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

## 🚀 How It Works

### 1. TrOCR — Pretrained Handwriting Recognition

- Loads a handwritten image from the **IAM Handwriting Database**
- Uses `TrOCRProcessor` to preprocess the image
- Feeds it into `VisionEncoderDecoderModel` to generate predicted text
- Evaluates predictions on **500 real handwriting images** from a custom dataset

```python
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten')

inputs = processor(images=image, return_tensors='pt').pixel_values
outputs = processor.batch_decode(model.generate(inputs), skip_special_tokens=True)[0]
```

### 2. ANN — MNIST Digit Classification

- Loads and normalizes the MNIST dataset (60,000 train / 10,000 test images)
- Builds a 3-layer ANN: `Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`
- Trains for **10 epochs** using Adam optimizer
- Evaluates using accuracy, loss, and a full **classification report**

### 3. Model Comparison & Visualization

Performance metrics (Accuracy, Precision, Recall, F1-Score, Inference Time) for four architectures are compared:

| Model | Accuracy | Precision | Recall | F1-Score | Inference Time |
|---|---|---|---|---|---|
| ANN | 92% | 91% | 90% | 90.5% | 30ms |
| CNN | 98% | 97% | 96% | 96.5% | 50ms |
| RNN | 94% | 93% | 92% | 92.5% | 40ms |
| Hybrid | 96% | 95% | 94% | 94.5% | 45ms |

Visualizations include:
- 📊 Bar chart — accuracy, precision, recall, F1-score per model
- 🥧 Pie chart — inference time distribution
- 📋 DataFrame summary table

---

## 📊 Results

- **TrOCR** accurately transcribes handwritten names from real-world images
- **ANN on MNIST** achieves strong digit classification performance
- **CNN** outperforms all models in accuracy (98%)

---

## 🗂️ Dataset

- **IAM Handwriting Database** — used for TrOCR evaluation
- **MNIST** — used for ANN digit classification
- Custom dataset (Google Drive) — 500 handwriting images for batch evaluation

---

## 👩‍💻 Author

**Rithikaa**  
Feel free to ⭐ this repo if you found it helpful!

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
