import random
import torch

def top_k(model, prompt, length, word_to_integer, integer_to_word, TOP_K):
    model.eval()
    hidden_state, cell_state = model.init_hidden(1)

    for word in prompt:
        word_tensor = torch.tensor([[word_to_integer[word]]])
        output, (hidden_state, cell_state) = model.forward(word_tensor, (hidden_state, cell_state))

    for i in range(length):
        _, distribution = torch.topk(output[0], k=TOP_K)
        choice = random.choice(list(distribution[0]))
        prompt.append(integer_to_word[choice.item()])

        word_tensor = torch.tensor([[choice]])
        output, (hidden_state, cell_state) = model.forward(word_tensor, (hidden_state, cell_state))

    return " ".join(prompt)


def get_raw_distribution(model, prompt, word_to_integer, integer_to_word):
    model.eval()
    hidden_state, cell_state = model.init_hidden(1)

    for word in prompt:
        word_tensor = torch.tensor([[word_to_integer[word]]])
        output, (hidden_state, cell_state) = model.forward(word_tensor, (hidden_state, cell_state))