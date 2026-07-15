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
 
# How it works:
- It randomizes the weights and biases
- Then defines the neural network class
- Once we make an instance of the network, we call train() and then print the results
- Inside of train(), there is a loop that runs for the amount of epochs. It first calculates the forward pass, with a2 being the final output of the model, then the loss is calculated, and then it calls self.backward(...) which manually derives W1, b1, W2, and b2 and returns their gradients. To adjust the parameters, it multiplies the gradients by the learning rate and subtract that from the parameter.

# Example Output:
```
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
