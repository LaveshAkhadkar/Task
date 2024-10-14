from transformers import BertTokenizer
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from streamlit import cache_resource

@cache_resource
def load_model():
    model = from_pretrained_keras("Lavesh-Akhadkar/hierarchical-bert-model")
    return model

@cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")

loaded_model = load_model()

classification_tokenizer = load_tokenizer()

def tokenize_data(text, max_len=128):
    return classification_tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="tf",
    )


def predict_with_loaded_model(text):
    encoding = tokenize_data(text)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    cat1_logits, cat2_logits = loaded_model((input_ids, attention_mask))

    cat1_pred = tf.argmax(cat1_logits, axis=-1).numpy()[0]
    cat2_pred = tf.argmax(cat2_logits, axis=-1).numpy()[0]

    return cat1_pred, cat2_pred
