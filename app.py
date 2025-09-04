import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer (make sure you push your trained model to Hugging Face Hub)
MODEL_NAME = "your-username/yelp-polarity-roberta"  # change to your repo after push

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

labels = ["Negative", "Positive"]

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Write a Yelp review here..."),
    outputs=gr.Label(num_top_classes=2),
    title="Yelp Polarity Classifier",
    description="A sentiment classifier trained on the Yelp Polarity dataset. Enter a review and get predicted sentiment."
)

if __name__ == "__main__":
    demo.launch()
