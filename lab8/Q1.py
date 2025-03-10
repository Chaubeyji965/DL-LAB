import os
import string
import glob
import unicodedata
import torch
import torch.nn as nn
import random


def find_files(path):
    return glob.glob(path)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def read_file(filename):
    category = os.path.splitext(os.path.basename(filename))[0]
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return category, [unicode_to_ascii(line) for line in lines]


category_lines = {}
categories = []

data_path = '/home/student/Desktop/220962422/lab8/data/names/*.txt'
for filename in find_files(data_path):
    category, lines = read_file(filename)
    category_lines[category] = lines
    categories.append(category)

n_categories = len(categories)


def letter_to_index(letter):
    return all_letters.find(letter)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

criterion = nn.NLLLoss()
learning_rate = 0.005



def random_choice(l):
    return l[random.randint(0, len(l) - 1)]



def random_training_example():
    category = random_choice(categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor



def train(category_tensor, line_tensor):
    hidden = rnn.init_hidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


n_iters = 100000
for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = random_training_example()
    output, loss = train(category_tensor, line_tensor)

    if iter % 5000 == 0:
        print(f'Iteration: {iter} Loss: {loss:.4f}')


torch.save(rnn.state_dict(), 'rnn_model.pth')

print('Training complete. Model saved.')
