from transformers import BartForConditionalGeneration, BartTokenizer
import torch

expansion_model = BartForConditionalGeneration.from_pretrained('Lavesh-Akhadkar/expander')
tokenizer = BartTokenizer.from_pretrained('Lavesh-Akhadkar/expanderTokenizer')

def generate_expanded_query(query, summary):
    input_text = f"{query} {summary}" if summary else query

    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)

    with torch.no_grad():
        outputs = expansion_model.generate(**inputs)

    expanded_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return expanded_query