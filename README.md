# yelp-polarity-roberta
# 🍔 Yelp Polarity Sentiment Classifier (RoBERTa-base)

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Demo-HuggingFace-blue)](https://huggingface.co/spaces/itserphan/Yelp-Polarity-Roberta)
[![Model on Hugging Face](https://img.shields.io/badge/🤗%20Model-Hub-orange)](https://huggingface.co/itserphan/yelp-polarity-roberta)

A fine-tuned **RoBERTa-base** model for **binary sentiment analysis** on the [Yelp Polarity dataset](https://huggingface.co/datasets/yelp_polarity).  
The model classifies reviews as **Positive** ⭐ or **Negative** 👎 with **96% accuracy**.

---

## 🚀 Features
- End-to-end pipeline for sentiment classification using Hugging Face `transformers`
- Preprocessing: lowercasing, tokenization, padding/truncation (`max_length=128`)
- Model: `roberta-base` fine-tuned with **Trainer API**
- Evaluation: Accuracy, Precision, Recall, F1
- Deployment: Gradio app + Hugging Face Space demo

---

## 📊 Results

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1 |
|-------|---------------|-----------------|----------|-----------|--------|----|
| 1     | 0.1224        | 0.1144          | 0.9595   | 0.9625    | 0.9563 | 0.9594 |
| 2     | 0.0946        | 0.1204          | 0.9613   | 0.9605    | 0.9622 | 0.9614 |
| 3     | 0.0735        | 0.1420          | 0.9618   | 0.9586    | 0.9654 | 0.9620 |
| 4     | 0.0514        | 0.1783          | 0.9620   | 0.9575    | 0.9670 | 0.9623 |
| 5     | 0.0353        | 0.2012          | 0.9623   | 0.9607    | 0.9640 | 0.9624 |

✅ Final Accuracy: **96.23%**  
✅ Balanced Precision & Recall → reliable across both classes  

---

## 🛠️ Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/itserphan/yelp-polarity-roberta.git
cd yelp-polarity-roberta
pip install -r requirements.txt
```

# Training
## Training from scratch with:
```
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

dataset = load_dataset("yelp_polarity")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, max_length=128)
tokenized = dataset.map(tokenize_fn, batched=True)

tokenized = tokenized["train"].train_test_split(test_size=0.2)

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# TrainingArguments + Trainer (see code in repo)
```


# Inference
## Use the fine-tuned model directly from Hugging Face Hub:

```
from transformers import pipeline

clf = pipeline("text-classification", model="itserphan/Yelp-Polarity-Roberta")

print(clf("The pizza was fantastic!")) 
# [{'label': 'Positive', 'score': 0.998}]

print(clf("Worst customer service ever.")) 
# [{'label': 'Negative', 'score': 0.997}]
```

# Demo App
## Try it live on Hugging Face Spaces 👉 [Demo here](https://huggingface.co/spaces/itserphan/Yelp-Polarity-Roberta)

# Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Datasets
* Gradio (for demo UI)

# Notes
* Base model: roberta-base
* Dataset: Yelp Polarity
* Training time (RTX 4050): ~40–60 mins for 5 epochs (batch size 32)
* Mixed precision (fp16) enabled for faster training

