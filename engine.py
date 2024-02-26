from model import ChessBot
import torch
import torch.nn as nn
import torch.nn.functional as F

num_layers = 15
model_size = 25
num_heads = 1
feed_forward_size = 15
board_size = (8, 8)  # Standard chessboard size
num_moves = 100  # Total possible moves in chess

chess_bot = ChessBot(num_layers, model_size, num_heads, feed_forward_size, board_size, num_moves)


from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data  # data is a list of tuples (board, move/outcome)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board, move = self.data[idx]
        # Convert board and move to tensor if not already
        return board, move


import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import time
def train(model, train_data, optimizer, criterion, device):
    model.train()
    total_loss = 0
    avg_time = 0

    for board, move in train_data:
        board, move = board.to(device), move.to(device)

        optimizer.zero_grad()

        t_1 = time.time()
        value, policy = model(board)
        t_2 = time.time()
        avg_time += t_2 - t_1
        
        # Assuming move is the correct policy distribution
        loss = criterion(value, move)  
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print("Time: ",avg_time / len(train_data))
    return total_loss / len(train_data)

def validate(model, val_data, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for board, move in val_data:
            board, move = board.to(device), move.to(device)

            value, policy = model(board)

            loss = criterion(value, move)
            total_loss += loss.item()

    return total_loss / len(val_data)









optimizer = Adam(chess_bot.parameters())
criterion =  nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = [
    (torch.randn(8, 8), torch.randn(1)), 
    (torch.randn(8, 8), torch.randn(1)),
    (torch.randn(8, 8), torch.randn(1)),
    (torch.randn(8, 8), torch.randn(1)),
    (torch.randn(8, 8), torch.randn(1)), 
    (torch.randn(8, 8), torch.randn(1)),
    (torch.randn(8, 8), torch.randn(1)),
    (torch.randn(8, 8), torch.randn(1)),
    (torch.randn(8, 8), torch.randn(1)), 
    (torch.randn(8, 8), torch.randn(1)),
    (torch.randn(8, 8), torch.randn(1)),
    (torch.randn(8, 8), torch.randn(1)),
    (torch.randn(8, 8), torch.randn(1)), 
    (torch.randn(8, 8), torch.randn(1)),
    (torch.randn(8, 8), torch.randn(1)),
    (torch.randn(8, 8), torch.randn(1)),
]

dataset = ChessDataset(data)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])


# Move model to device
chess_bot.to(device)
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    train_loss = train(chess_bot, train_data, optimizer, criterion, device)
    val_loss = validate(chess_bot, val_data, criterion, device)

    # Print or log the losses
    print(f'Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

# save model
torch.save(chess_bot.state_dict(), 'chessbot.pth')