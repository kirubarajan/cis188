import torch
from models import Model
from data import CorpusDataset
from decoding import top_k, get_raw_distribution

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

# instantiating dataset, model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CorpusDataset(TRAIN_PATH, CHUNK_SIZE, BATCH_SIZE)
model = Model(EMBEDDING_DIM, HIDDEN_DIM, len(dataset.vocabulary), device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

iteration = 0

for i in range(NUM_EPOCHS):
    hidden_state, cell_state = model.init_hidden(BATCH_SIZE)
    
    for text, target in dataset:
        model.train()
        optimizer.zero_grad()

        output, (hidden_state, cell_state) = model(text, (hidden_state, cell_state))
        loss = criterion(output.transpose(1, 2), target).to(device)

        hidden_state, cell_state = hidden_state.detach(), cell_state.detach()
        
        # perform gradient descent step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM)
        optimizer.step()

        # print loss and generate sample
        iteration += 1

        if iteration % 50 == 0: 
            print('Epoch: {}/{}'.format(i, NUM_EPOCHS), 'Iteration: {}'.format(iteration), 'Loss: {}'.format(loss.item()))
            # print(top_k(model, ["I", "am"], 100, dataset.word_to_integer, dataset.integer_to_word, TOP_K))

print("Model is finished training.")
prompt = "Elle est très".split(" ")

torch.save(model, 'models/french.pt')