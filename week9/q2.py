import os
import glob
import unicodedata
import string
import torch
import torch.nn as nn
import torch.optim as optim
import random

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )


def load_data(data_path):
    category_lines = {}
    all_categories = []
    if not os.path.exists(data_path):
        print(f"Data path '{data_path}' does not exist.")
        return category_lines, all_categories
    file_list = glob.glob(os.path.join(data_path, '*.txt'))
    if not file_list:
        print(f"No '.txt' files found in '{data_path}'.")
        return category_lines, all_categories
    for filename in file_list:
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        with open(filename, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            lines = [unicodeToAscii(line) for line in lines if line.strip()]
        category_lines[category] = lines
    return category_lines, all_categories


data_path = "/home/student/PycharmProjects/PythonProject2/data/names"
category_lines, all_categories = load_data(data_path)

n_categories = len(all_categories)


def letterToIndex(letter):
    return all_letters.find(letter)


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        index = letterToIndex(letter)
        if index != -1:
            tensor[li][0][index] = 1
    return tensor


def randomTrainingExample():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, line_tensor, category_tensor


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return self.softmax(output)


n_hidden = 128
lstm = LSTMClassifier(n_letters, n_hidden, n_categories)
criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = optim.SGD(lstm.parameters(), lr=learning_rate)


def train(name_tensor, category_tensor):
    lstm.zero_grad()
    name_tensor = name_tensor.view(1, name_tensor.size(0), -1)
    output = lstm(name_tensor)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return loss.item()


n_iters = 100000
print_every = 5000
current_loss = 0

for iter in range(1, n_iters + 1):
    category, line, line_tensor, category_tensor = randomTrainingExample()
    loss = train(line_tensor, category_tensor)
    current_loss += loss
    if iter % print_every == 0:
        avg_loss = current_loss / print_every
        current_loss = 0
        print(f"Iteration {iter} Loss: {avg_loss:.4f}")


def evaluate(line_tensor):
    line_tensor = line_tensor.view(1, line_tensor.size(0), -1)
    output = lstm(line_tensor)
    return output


def predict(input_line, n_predictions=1):
    print(f"\n> {input_line}")
    output = evaluate(lineToTensor(input_line))
    topv, topi = output.topk(n_predictions)
    predictions = []
    for i in range(n_predictions):
        value = topv[0][i].item()
        category_index = topi[0][i].item()
        language = all_categories[category_index]
        print(f"({value:.4f}) {language}")
        predictions.append((value, language))
    return predictions


predict("Dostoevsky")
predict("Sokolov")
predict("Schmidt")