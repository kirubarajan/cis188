from flask import Flask, render_template, request, redirect, url_for
from models import Model
from decoding import top_k

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
torch.load(model, 'models/french.pt')


app = Flask(__name__)


@app.route("/predict")
def splash():
    data = request.get_json()
    input_sequence = data['input']
    output_sequence = top_k(model, input_sequence.split(), 25, dataset.word_to_integer, dataset.integer_to_word, TOP_K)

    return {'prediction': output_sequence}


if __name__ == "__main__":
    app.run(port=8000, host='0.0.0.0', debug=True)