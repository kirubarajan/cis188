import torch

class CorpusDataset(torch.utils.data.IterableDataset):
    def __init__(self, FILE_PATH, CHUNK_SIZE, BATCH_SIZE):
        super(CorpusDataset).__init__()
        self.CHUNK_SIZE, self.BATCH_SIZE = CHUNK_SIZE, BATCH_SIZE

        with open(FILE_PATH) as file:
            text = file.read().split()
            self.vocabulary = set(text)

        ordered_vocab = sorted(list(self.vocabulary))
        self.integer_to_word = {integer: word for integer, word in enumerate(ordered_vocab)}
        self.word_to_integer = {word: integer for integer, word in enumerate(ordered_vocab)}

        self.num_batches = int(len(text) / (CHUNK_SIZE * BATCH_SIZE))
        encoded_text = [self.word_to_integer[word] for word in text][:self.num_batches * BATCH_SIZE * CHUNK_SIZE]
        target_text = encoded_text[1:] + [encoded_text[0]]

        self.encoded_text = torch.tensor(encoded_text).view(BATCH_SIZE, -1)
        self.target_text = torch.tensor(target_text).view(BATCH_SIZE, -1)

    def __iter__(self):
        for i in range(0, self.num_batches * self.CHUNK_SIZE, self.CHUNK_SIZE):
            yield self.encoded_text[:, i:i + self.CHUNK_SIZE], self.target_text[:, i:i + self.CHUNK_SIZE]