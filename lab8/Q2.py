import torch
import torch.nn as nn
import torch.optim as optim


text = input("Enter the text to train the model: ").lower()


chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


input_size = len(chars)
hidden_size = 128
output_size = len(chars)
sequence_length = 10
learning_rate = 0.005
num_epochs = 500



def create_sequences(text, seq_length):
    inputs = []
    targets = []
    for i in range(len(text) - seq_length):
        inputs.append(text[i:i + seq_length])
        targets.append(text[i + seq_length])
    return inputs, targets


inputs, targets = create_sequences(text, sequence_length)



def text_to_tensor(text):
    return torch.tensor([char_to_idx[c] for c in text])


input_tensors = torch.stack([text_to_tensor(seq) for seq in inputs])
target_tensors = torch.tensor([char_to_idx[t] for t in targets])



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden



model = RNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    hidden = torch.zeros(1, input_tensors.size(0), hidden_size)

    outputs, hidden = model(input_tensors, hidden)
    loss = criterion(outputs, target_tensors)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')



def predict_next_char(model, start_text, length):
    model.eval()
    hidden = torch.zeros(1, 1, hidden_size)
    input_text = start_text.lower()

    for _ in range(length):

        input_tensor = text_to_tensor(input_text[-sequence_length:]).unsqueeze(0)
        output, hidden = model(input_tensor, hidden)
        predicted_idx = torch.argmax(output, dim=1).item()
        predicted_char = idx_to_char[predicted_idx]
        input_text += predicted_char

    return input_text



start_text = input("\nEnter the starting text: ")
length = int(input("Enter the number of characters to generate: "))


generated_text = predict_next_char(model, start_text, length)
print("\nGenerated Text:")
print(generated_text)
