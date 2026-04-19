# Self-Pruning Neural Network

This project builds a neural network that prunes itself during training.

Each weight has a gate (0 to 1). L1 loss pushes many gates to 0 → making model sparse.

Output:
- Sparsity %
- results.png (gate distribution)