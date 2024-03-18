# libraries 

from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
def QAModel():
    model_name = "distilbert-base-cased-distilled-squad"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = TFDistilBertForQuestionAnswering.from_pretrained(model_name)
    return model_name, tokenizer, model


model=QAModel()
