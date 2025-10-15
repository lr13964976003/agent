## Tensor Parallelism

### Fundamental Concept

Tensor parallelism splits individual layers (matrices) across multiple devices, allowing parallel computation of matrix operations. This is particularly effective for large linear layers in transformers.

### Matrix Multiplication Partitioning

#### Row Parallel Linear Layer
```
Input: X (batch_size, input_dim)
Weight: W (output_dim, input_dim)
Output: Y = XW^T (batch_size, output_dim)

Partitioning:
- Device 0: W_0 (output_dim/2, input_dim)
- Device 1: W_1 (output_dim/2, input_dim)

Forward Pass:
Y_0 = XW_0^T  # On device 0
Y_1 = XW_1^T  # On device 1
Y = [Y_0; Y_1]  # Concatenation across devices
```

#### Column Parallel Linear Layer
```
Input: X (batch_size, input_dim)
Weight: W (output_dim, input_dim)
Output: Y = XW^T (batch_size, output_dim)

Partitioning:
- Device 0: W_0 (output_dim, input_dim/2)
- Device 1: W_1 (output_dim, input_dim/2)

Forward Pass:
X_0 = X[:, :input_dim/2]  # Split input
X_1 = X[:, input_dim/2:]
Y_0 = X_0 W_0^T  # On device 0
Y_1 = X_1 W_1^T  # On device 1
Y = Y_0 + Y_1    # All-reduce sum
```

#### MLP Layer Tensor Parallel

MLP tensor parallelism combines column and row parallel strategies to parallelize the two linear layers in a transformer MLP block efficiently.

```
MLP Structure:
- First Linear: hidden_size → ffn_hidden_size  (Column-parallel)
- Activation: GELU
- Second Linear: ffn_hidden_size → hidden_size  (Row-parallel)

Partitioning Strategy:
- Device 0: First layer W1_0 (ffn_hidden_size/2, hidden_size), Second layer W2_0 (hidden_size, ffn_hidden_size/2)
- Device 1: First layer W1_1 (ffn_hidden_size/2, hidden_size), Second layer W2_1 (hidden_size, ffn_hidden_size/2)

Forward Pass:
1. First linear (column-parallel):
   - Input X (batch_size, hidden_size) is broadcast to all devices
   - intermediate_0 = X W1_0^T  # On device 0
   - intermediate_1 = X W1_1^T  # On device 1
   - intermediate = [intermediate_0; intermediate_1]  # Concatenation

2. Activation function:
   - intermediate = GELU(intermediate)  # Applied element-wise

3. Second linear (row-parallel):
   - intermediate_0 = intermediate[:, :ffn_hidden_size/2]  # Split along feature dim
   - intermediate_1 = intermediate[:, ffn_hidden_size/2:]
   - output_0 = intermediate_0 W2_0^T  # On device 0
   - output_1 = intermediate_1 W2_1^T  # On device 1
   - output = output_0 + output_1  # All-reduce sum across devices
```

