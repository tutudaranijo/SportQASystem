# libraries 

from transformers import TFBertModel, BertTokenizer
def QAModel():
    model_name = 'bert-base-uncased'  
    bert_model = TFBertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model_name, bert_model, tokenizer


