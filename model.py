import torch
import torch.nn as nn

class DigitalBodyguard(nn.Module):
    # Notice we added pretrained_embeddings here, and changed embed_dim to 100 to match GloVe
    def __init__(self, vocab_size, pretrained_embeddings=None, embed_dim=100, hidden_dim=256, output_dim=3, dropout=0.3):
        super(DigitalBodyguard, self).__init__()
        
        # --- NEW V2 CODE: THE SMART EMBEDDING ---
        if pretrained_embeddings is not None:
            # Load the GloVe knowledge! 
            # freeze=False means the AI can still tweak the meanings slightly as it learns gamer slang
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            # Fallback just in case
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
            
        # ... the rest of your LSTM and Linear layers stay EXACTLY the same ...
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, 
                            num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, text_tensors):
        # text_tensors shape: [batch_size, sequence_length] -> [64, 788]
        
        # Pass through embedding
        embedded = self.embedding(text_tensors)
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Extract the very last hidden state from both the forward and backward passes
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        
        # Glue them together
        final_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        # Get the final 3 predictions
        final_hidden = self.dropout(final_hidden)
        predictions = self.fc(final_hidden)
        
        return predictions

# --- PERSON B's HANDSHAKE TEST ---
if __name__ == "__main__":
    print("Booting up the Brain Architecture...")
    
    # We use the EXACT numbers Person A reported from their data pipeline
    mock_vocab_size = 144419
    model = DigitalBodyguard(vocab_size=mock_vocab_size)
    
    # Create a fake batch of data matching Person A's [64, 788] output
    mock_input = torch.randint(0, mock_vocab_size, (64, 788))
    
    print(f"Feeding fake tensor of shape: {mock_input.shape}")
    mock_output = model(mock_input)
    
    print(f"Model spit out tensor of shape: {mock_output.shape}")
    if mock_output.shape == torch.Size([64, 3]):
        print("PERSON B HANDSHAKE SUCCESS! 🤝 The brain is perfectly sized.")
    else:
        print("CRITICAL FAILURE: Tensor shape mismatch.")