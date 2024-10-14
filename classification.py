from transformers import BertTokenizer
import tensorflow as tf
from huggingface_hub import from_pretrained_keras

from transformers import TFBertModel
import tensorflow as tf
from huggingface_hub import KerasModelHubMixin

class HierarchicalBERTModel(tf.keras.Model, KerasModelHubMixin):
    def __init__(self, num_cat1=6, num_cat2=21, bert_model_name='bert-base-uncased'):
        super(HierarchicalBERTModel, self).__init__()
        self.bert = TFBertModel.from_pretrained(bert_model_name)
        self.cat1_classifier = tf.keras.layers.Dense(num_cat1, activation='softmax')
        self.cat2_classifier = tf.keras.layers.Dense(num_cat2, activation='softmax')

    def call(self, inputs):
        # Allow inputs in both tuple and dict format
        if isinstance(inputs, dict):
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
        else:
            input_ids, attention_mask = inputs
        bert_output = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_output = bert_output.pooler_output

        cat1_logits = self.cat1_classifier(cls_token_output)
        cat2_logits = self.cat2_classifier(cls_token_output)

        return cat1_logits, cat2_logits



loaded_model = HierarchicalBERTModel()

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
