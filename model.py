import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, feature_size):
        super(LayerNorm, self).__init__()
        # Initialize weights and bias
        self.layer_norm = nn.LayerNorm(feature_size)

    def forward(self, x):
        # print(x.shape)
        # print(self.layer_norm(x).shape)
        return self.layer_norm(x)

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()
        # print(input_size, hidden_size, output_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.ln1 = LayerNorm(hidden_size)
        self.ln2 = LayerNorm(output_size)

    def forward(self, x):
        # Apply first layer transformation
        # print(x.shape)
        hidden = self.linear1(x)
        hidden = self.ln1(hidden)
        hidden = F.relu(hidden)

        # Apply second layer transformation
        output = self.linear2(hidden)
        output = self.ln2(output)

        return output


class SelfAttention(nn.Module):
    def __init__(self, num_heads, model_size):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.model_size = model_size
        self.head_dim = model_size // num_heads

        assert self.head_dim * num_heads == model_size, "model_size must be divisible by num_heads"

        self.query = nn.Linear(model_size, model_size)
        self.key = nn.Linear(model_size, model_size)
        self.value = nn.Linear(model_size, model_size)

    def forward(self, x):
        print(x.shape)

        # Calculate query, key, value matrices
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Split the matrices into heads and reshape
        # queries = self.split_heads(queries, batch_size)
        # keys = self.split_heads(keys, batch_size)
        # values = self.split_heads(values, batch_size)

        # Scaled Dot-Product Attention
        attention = self.scaled_dot_product_attention(queries, keys, values)

        # Concatenate heads and apply final linear transformation
        # attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.model_size)

        return attention

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def scaled_dot_product_attention(self, queries, keys, values):
        scaling_factor = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        # attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / scaling_factor
        # keys = keys.transpose(0,1)
        # print(keys.shape, queries.shape)
        # transpose 25* 1 keys to 1 * 25
        print(queries.shape, keys.shape, values.shape)
        attention_scores = torch.matmul(queries.view(-1, 1), keys.view(1, -1)) / scaling_factor
        print(attention_scores.shape, queries.view(-1, 1).shape, keys.view(1, -1).shape)
        # print(attention_scores)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, values)
        print("output shape", output.shape)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, model_size, num_heads, feed_forward_size):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(num_heads, model_size)
        self.norm1 = LayerNorm(model_size)
        self.norm2 = LayerNorm(model_size)
        self.feed_forward = FeedForward(model_size, feed_forward_size, model_size)

    def forward(self, x):
        # Step 1: Self-Attention
        attention_output = self.attention(x)

        # Step 2: Add & Norm (Residual connection + LayerNorm)
        x = self.norm1(x + attention_output)

        # Step 3: Feed-Forward
        feed_forward_output = self.feed_forward(x)

        # Step 4: Add & Norm (Another residual connection + LayerNorm)
        x = self.norm2(x + feed_forward_output)

        return x


class Transformer(nn.Module):
    def __init__(self, num_layers, model_size, num_heads, feed_forward_size):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(model_size, num_heads, feed_forward_size) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class ChessBot(nn.Module):
    def __init__(self, num_layers, model_size, num_heads, feed_forward_size, board_size, num_moves):
        super(ChessBot, self).__init__()
        self.transformer = Transformer(num_layers, model_size, num_heads, feed_forward_size)
        self.inputff = FeedForward(board_size[0] * board_size[1], feed_forward_size, model_size)

        # Value head: A simple linear layer that outputs a scalar
        self.value_head = nn.Linear(model_size, 1)

        # Policy head: A linear layer that outputs a probability distribution over moves
        self.policy_head = nn.Linear(model_size, num_moves)

        # Assuming board_size is a tuple like (8, 8) for a standard chessboard
        self.board_size = board_size
        self.input_size = board_size[0] * board_size[1]

    def forward(self, x):
        # x is the multi-channel grid input of shape [batch, channels, height, width]

        # Flatten the board representation to fit the transformer input
        # print(x.shape)
        x = x.view(-1)  # Flatten to [batch, channels * height * width]
        # print(x.shape)
        
        x = self.inputff(x)

        # Apply Transformer layers
        # x = torch.tensor 
        x = self.transformer(x)

        # Select a representative vector (e.g., the first one) for heads
        # x = x[:, 0]

        # Value head
        value = torch.tanh(self.value_head(x))

        # Policy head
        policy = self.policy_head(x)
        policy = F.softmax(policy, dim=-1)

        return value, policy

