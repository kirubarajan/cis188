import torch
from flask import Flask, render_template, request, redirect, url_for
from models import Model
from data import CorpusDataset
from decoding import top_k
from bleu import list_bleu

# defining hyperparameters
NUM_EPOCHS = 1
LEARNING_RATE = 0.01
GRADIENT_NORM = 5
EMBEDDING_DIM = 32
TOP_K = 5
HIDDEN_DIM = 32
BATCH_SIZE = 16
CHUNK_SIZE = 32
TRAIN_PATH = "data/french.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CorpusDataset(TRAIN_PATH, CHUNK_SIZE, BATCH_SIZE)
model = Model(EMBEDDING_DIM, HIDDEN_DIM, len(dataset.vocabulary), device)

model.load_state_dict(torch.load('models/french.pt'))
model.eval()
input_sequence = "Le"

# running BLEU evaluation
ref = [top_k(model, input_sequence.split(), 25, dataset.word_to_integer, dataset.integer_to_word, TOP_K, sample=True)]
hyp = ['Le comportement de la', ]

output = list_bleu([ref], hyp)
with open('results.txt', 'w+') as f:
    f.write("BLEU Metric: " + str(output))