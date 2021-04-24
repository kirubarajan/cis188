"""# Inference and Usage"""
import torch
import torch.nn as nn
import spacy

nlp = spacy.load('en')

model.load_state_dict(torch.load('model.jawn'))

def predict_sentiment(model, sentence):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

# predict_sentiment(model, "i hate this movie a lot and it was really bad")