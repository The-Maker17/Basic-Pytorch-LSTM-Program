import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().split()

file_path = 'data.txt'
words = load_text(file_path)
vocab = list(set(words)) + ["<UNK>"]
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
sequence_length = 10
data = [(words[i:i + sequence_length], words[i + sequence_length]) for i in range(len(words) - sequence_length)]

class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

vocab_size, embedding_dim, hidden_dim = len(vocab), 10, 20
model = SimpleModel(vocab_size, embedding_dim, hidden_dim)

def encode(sequence, word_to_idx, unk_token="<UNK>"):
    return torch.tensor([word_to_idx.get(word, word_to_idx[unk_token]) for word in sequence], dtype=torch.long)

X = [encode(seq, word_to_idx) for seq, _ in data]
Y = [word_to_idx[target] for _, target in data]

criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001)
for epoch in range(10):
    total_loss = 0
    for i in range(len(X)):
        inputs = X[i].unsqueeze(0)
        target = torch.tensor([Y[i]], dtype=torch.long)
        output = model(inputs)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(X):.4f}")

torch.save(model.state_dict(), 'model.pth')

def generate_text(model, start_seq, word_to_idx, idx_to_word, length=20):
    seq = start_seq.copy()
    for _ in range(length):
        input_indices = encode(seq[-sequence_length:], word_to_idx).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(input_indices), dim=-1)
        seq.append(idx_to_word[torch.argmax(probs).item()])
    return ' '.join(seq)

def chatbot(model, word_to_idx, idx_to_word):
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        start_seq = [word if word in word_to_idx else "<UNK>" for word in user_input.split()]
        response = generate_text(model, start_seq, word_to_idx, idx_to_word)
        print(f"Chatbot: {response}")

model.eval()
chatbot(model, word_to_idx, idx_to_word)
