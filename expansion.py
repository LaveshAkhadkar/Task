from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from streamlit import cache_resource

@cache_resource
def load_expansion_model():
    return BartForConditionalGeneration.from_pretrained('Lavesh-Akhadkar/expander')

@cache_resource
def load_tokenizer():
    return BartTokenizer.from_pretrained('Lavesh-Akhadkar/expanderTokenizer')

expansion_model = load_expansion_model()
tokenizer = load_tokenizer()

def generate_expanded_query(query, summary):
    input_text = f"{query} {summary}" if summary else query

    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)

    with torch.no_grad():
        outputs = expansion_model.generate(**inputs)

    expanded_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return expanded_query
