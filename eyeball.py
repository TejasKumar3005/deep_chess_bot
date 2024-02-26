from model import ChessBot

import torch

num_layers = 15
model_size = 25
num_heads = 1
feed_forward_size = 15
board_size = (8, 8)  # Standard chessboard size
num_moves = 100  # Total possible moves in chess

model = ChessBot(num_layers, model_size, num_heads, feed_forward_size, board_size, num_moves)  # Replace with your model class
model.load_state_dict(torch.load('chessbot.pth'))

# Print model parameters
def write_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")

    if tensor.dim() == 2:
        print("{", end="")
        for row in tensor:
            print("{", end="")
            for i, value in enumerate(row):
                if i < len(row) - 1:
                    print(f"{float(value):.1f}, ", end="")
                else:
                    print(f"{float(value):.1f}", end="")
            print("},")
        print("}")
    elif tensor.dim() == 1:
        print("{", end="")
        for i, value in enumerate(tensor):
            if i < len(tensor) - 1:
                print(f"{float(value):.1f}, ", end="")
            else:
                print(f"{float(value):.1f}", end="")
        print("}")
    else:
        raise ValueError("Input tensor must have 1 or 2 dimensions")


    
for name, param in model.named_parameters():
    # replace . with _ in name
    name = name.replace('.', '_')
    print("Matrix ",name, " ", "(")
    write_tensor(param)
    print(");")    
            
    
    
    

# tell number of parameters
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))