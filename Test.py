from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import torch as th
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchtext
from MyRNN import RNN

device = "cpu"#"mps" if th.backends.mps.is_available() else "cpu"


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=1):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = RNN(input_size=embed_size, hidden_size=hidden_size, output_size=hidden_size, num_layer=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        embedded = self.embedding(text)
        out, _ = self.rnn(embedded)
        out = self.dropout(out[-1])
        return self.fc(out)
    

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

train_iter = IMDB(split="train")
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

labels = {'pos': 1, 'neg': 0}

def collate_batch(batch):
    label_list, text_list = [], []
    for (label, text) in batch:
        label_list.append(label - 1)
        processed_text = th.tensor(vocab(tokenizer(text)), dtype=th.long)
        text_list.append(processed_text)

    text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=False, padding_value=vocab["<pad>"]).to(device)
    label_list = th.tensor(label_list, dtype=th.long).to(device)
    return text_list, label_list


train_dataset = list(IMDB(split="train"))[:1000]
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_batch)
vocab_size = len(vocab)
embed_size = 300
hidden_size = 300
num_classes = 2
num_layers = 2
lr = 1e-3
epochs = 5

model = SentimentRNN(vocab_size, embed_size, hidden_size, num_classes, num_layers)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")

model.train()
for epoch in tqdm(range(epochs)):
    epoch_loss = 0.0
    for text_batch, label_batch in train_loader:
        optimizer.zero_grad()
        logits = model(text_batch)
        loss = criterion(logits, label_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.12f}")


th.save(model, "Model")

def predict(sentence):
    model.eval()
    tokens = tokenizer(sentence)
    indices = [vocab[token] for token in tokens]
    input_tensor = th.tensor(indices, dtype=th.long).unsqueeze(1).to(device)
    with th.no_grad():
        logits = model(input_tensor)

    probabilities = th.softmax(logits, dim=1).squeeze()
    predicted_label = probabilities.argmax().item()

    label_mapping = {0: "Negative", 1: "Positive"}
    predicted_sentiment = label_mapping[predicted_label]

    return predicted_sentiment, probabilities


running = True

while running:
    input_ = input("-> ")
    if input_ == "q":
        running = False
    else:
        print(predict(input_))

