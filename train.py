import torch
import torch.nn as nn
import torch.optim as optim
from dataset import prepare_dataloaders
from model import DigitalBodyguard

def run_training_engine():
    print("1. Igniting the Data Pipeline...")
    
    # V2 UPDATE: We now catch the 4th item (pretrained_embeddings) from the pipeline!
    train_loader, val_loader, vocab, pretrained_embeddings = prepare_dataloaders("clean_data.csv", batch_size=16)
    vocab_size = len(vocab)
    print(f"Vocab Size loaded: {vocab_size}")

    print("2. Detecting Hardware...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device.type.upper()}")

    print("3. Configuring Loss & Optimizer...")
    # Less paranoid weights so it doesn't flag "good morning"
    class_weights = torch.tensor([1.0, 1.5, 2.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\n--- DAY 2: THE DEEP BAKE (10 EPOCHS) ---")
    
    # V2 UPDATE: Initialize the model with the GloVe Knowledge Matrix!
    # Notice we pass pretrained_embeddings and changed embed_dim to 100
    model = DigitalBodyguard(
        vocab_size=vocab_size, 
        pretrained_embeddings=pretrained_embeddings, 
        embed_dim=100, 
        hidden_dim=256
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10 
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        print(f"\nEpoch {epoch+1}/{epochs} starting...")
        
        for batch_idx, (texts, labels) in enumerate(train_loader):
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(texts)
            loss = criterion(predictions, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Print an update every 2000 batches so it doesn't spam your terminal too much
            if batch_idx % 2000 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)} | Current Loss: {loss.item():.4f}")
            
        print(f"-> Epoch {epoch+1} Completed | Average Loss: {total_loss/len(train_loader):.4f}")
        
    print("\n--- TRAINING COMPLETE ---")
    print("Saving the Bodyguard's Brain...")
    
    # Save the trained weights to a file!
    torch.save(model.state_dict(), "bodyguard_weights.pt")
    torch.save(vocab, "bodyguard_vocab.pt")
    print("Brain saved successfully as 'bodyguard_weights.pt' and 'bodyguard_vocab.pt'!")

if __name__ == "__main__":
    run_training_engine()