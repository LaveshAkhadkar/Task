from transformers import BertTokenizer
import tensorflow as tf
from huggingface_hub import from_pretrained_keras

loaded_model.built = True
loaded_model = from_pretrained_keras("Lavesh-Akhadkar/hierarchical-bert-model")

dummy_input_ids = tf.zeros((1, 128), dtype=tf.int32)
dummy_attention_mask = tf.zeros((1, 128), dtype=tf.int32)

loaded_model((dummy_input_ids, dummy_attention_mask))

dummy_input_ids = tf.zeros((1, 128), dtype=tf.int32)
dummy_attention_mask = tf.zeros((1, 128), dtype=tf.int32)

loaded_model((dummy_input_ids, dummy_attention_mask))

classification_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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
