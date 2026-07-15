# XOR-Gate-Neural-Net-Using-Numpy
It's been a while since publishing my last repo! I spent a while learning about the math needed for moving from my scalar based logic in C++ to tensor based logic using NumPy, and along with that I learned a lot as a whole about linear algebra along with some multivariable calculus and differential calculus.

# Features:
- XOR dataset
- Dynamic layer sizes
- Forward pass
- MSE loss
- Manual backpropagation

# Architecture (base code):
- Input size: 2
- Hidden size: 4
- Output size: 1
 
# How It Works
1. Initialization: Randomizes weights and biases for all layers.
2. Definition: Creates the core structural neural network class.
3. Training: Runs a training loop for a set number of epochs.
   * Calculates the forward pass to get the final output (`a2`).
   * Computes the Mean Squared Error (MSE) loss.
   * Calls `self.backward()` to manually derive the gradients for `W1`, `b1`, `W2`, and `b2`.
   * Updates parameters by subtracting the gradients multiplied by the learning rate.

# Example Output:
```text
0.24947201006750402
0.1297168294397139
0.12953097206021658
0.1288808001773607
0.1266337357249303
0.11919266932638518
0.09799293338538745
0.05947200938488688
0.028112049334212214
0.0005166681838534471
[[0.005]
 [0.99 ]
 [0.997]
 [0.027]]
```
