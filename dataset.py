import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, GloVe
from sklearn.model_selection import train_test_split

# 1. Custom Dataset Class
class BodyguardDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts.iloc[idx], self.labels.iloc[idx]

def prepare_dataloaders(csv_path="clean_data.csv", batch_size=64):
    print("1. Loading cleaned data...")
    df = pd.read_csv(csv_path)
    
    # Split: 80% Train, 10% Validation, 10% Test
    X_train, X_temp, y_train, y_temp = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("2. Building vocabulary (This takes a moment for 300k+ rows)...")
    tokenizer = get_tokenizer('basic_english')
    
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(str(text))

    # Build vocabulary with special tokens
    vocab = build_vocab_from_iterator(yield_tokens(X_train), specials=["<pad>", "<unk>"], min_freq=2)
    vocab.set_default_index(vocab["<unk>"])

    # --- V2 CODE: THE GLOVE INJECTION ---
    print("Loading GloVe Dictionary...")
    glove = GloVe(name='6B', dim=100)
    
    embed_dim = 100
    vocab_size = len(vocab)
    pretrained_embeddings = torch.zeros(vocab_size, embed_dim)
    
    words_found = 0
    for i, word in enumerate(vocab.get_itos()): 
        if word in glove.stoi:
            pretrained_embeddings[i] = glove[word]
            words_found += 1
        else:
            pretrained_embeddings[i] = torch.randn(embed_dim)

    print(f"GloVe mapped {words_found} out of {vocab_size} words perfectly!")
    # ------------------------------------

    # 3. Text Pipeline
    def text_pipeline(text):
        return vocab(tokenizer(str(text)))

    # 4. Collate Function (Padding)
    def collate_batch(batch):
        label_list, text_list = [], []
        for (_text, _label) in batch:
            label_list.append(_label)
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
        
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = pad_sequence(text_list, padding_value=vocab["<pad>"], batch_first=True)
        return text_list, label_list

    print("3. Initializing DataLoaders...")
    # ---> THIS IS THE CODE THAT GOT DELETED ACCIDENTALLY <---
    train_dataset = BodyguardDataset(X_train, y_train)
    val_dataset = BodyguardDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    # Return all 4 items!
    return train_loader, val_loader, vocab, pretrained_embeddings

if __name__ == "__main__":
    train_loader, val_loader, vocab, pretrained_embeddings = prepare_dataloaders()
    
    print("\n--- The Handshake Test ---")
    print(f"Total Unique Words in Vocab: {len(vocab)}")
    print(f"GloVe Matrix Shape: {pretrained_embeddings.shape}")