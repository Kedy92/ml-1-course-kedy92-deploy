In Lesson 4, we built an MLP (multilayer perceptron), sometimes also called ANN, artifical neural network, that classified Titanic passengers. We wrote the model, called `loss.backward()`, and watched accuracy climb. It worked. But we treated most of it as a black box - layers were stacked, a training loop ran, numbers improved. We didn't stop to ask what the numbers actually look like as they flow through the network, or why `loss.backward()` magically knows which weights to fix.

This lesson opens the black box. We trace real data through a neural network step by step - watching matrix multiplications happen, seeing what ReLU does to actual values, understanding how the loss tells us exactly how wrong we are. Then we follow the signal backwards: how gradients assign blame to each weight, and how optimizers use that information to make the network better. By the end, you'll understand every moving part of the training loop you've already been using.

The training loop pattern (forward - loss - backward - update - repeat) is the same whether you're training on 712 Titanic passengers or 1.2 million images. Once you understand it here, you understand it everywhere.

One thing that took me a while to grasp as a beginner: **the training loop IS gradient descent.** In L3, we learned gradient descent as "compute the gradient, take a step downhill, repeat." That's exactly what the training loop does - it's just spread across four named steps. The forward pass computes predictions. The loss measures error. The backward pass computes gradients (one per weight - a matrix of partial derivatives saying how much each weight contributed to the error). The weight update is the actual "step downhill." That's gradient descent. When you see `loss.backward()` followed by `optimizer.step()`, that IS the same algorithm from L3 - just applied to 97 weights through multiple layers instead of 2 weights in a straight line.

## Key Concepts

**The Forward Pass** is where data flows through the network. This part usually takes some time to understand because it involves multiplying each row with weights, which in turn ends up creating new inputs for upcoming layers in the neural network. It’s like a step by step process, where finally, by the end, we get some outputs that are the probabilities for our problem - whether its regressio nor classifation. Each layer does the same thing: multiply inputs by weights, add a bias, apply an activation function. That's it. For our Titanic MLP, passenger features (Pclass, Sex, Age, Fare) enter as a tensor, get multiplied by a weight matrix in the first linear layer, pass through ReLU (which zeros out negatives and keeps positives), then hit a second linear layer and finally sigmoid to produce a probability. We trace this with actual numbers in the notebook - you'll see exactly what a 4-feature input looks like after each transformation.

**Matrix Multiplication** is how neural networks process data efficiently. Instead of computing one neuron at a time, we multiply the entire input batch against the weight matrix in a single operation. Each row in the weight matrix is one neuron's "recipe" for combining inputs. The shape rule is simple: `(samples x features) @ (features x neurons) = (samples x neurons)`. This is why neural networks run fast on GPUs - matrix math parallelizes beautifully.

**Loss Calculation** measures how wrong the predictions are. For binary classification, we use binary cross-entropy. The key intuition: the model gets penalized more for confident wrong answers than uncertain ones. Predicting 0.99 when the answer is 0 is much worse than predicting 0.6 when the answer is 0. This asymmetric punishment is what drives the network to be both accurate and calibrated. Depending on the task, we use different loss functions: `BCELoss` (binary cross-entropy) for binary classification like Titanic, `CrossEntropyLoss` for multi-class classification like digit recognition (0-9), or `MSELoss` for regression like house prices. You’ll see all three as the course progresses.

**Backpropagation** answers the question: "which weights caused the error, and by how much?" When you call `loss.backward()`, PyTorch traces backwards through every operation (sigmoid, matrix multiply, ReLU, another matrix multiply) and computes a gradient for each weight. Think of it as blame assignment - each weight gets a number saying "you contributed this much to the loss." You don't need to compute derivatives by hand. PyTorch's autograd system tracks every operation and applies the chain rule automatically. But understanding that this process exists, and that it produces one gradient per weight, is essential. But it’s not critical to understand exactly how it works, e.g the chain rule - backprop is actually one of the hardest things to grasp about neural networks, and a large part of that is because the math is based on partial derivatives… so we skip that ;-)

**The Weight Update** is what learning actually is. Once we have gradients (imagine a matrix of numbers that we use to update the weights, which also happens to be a matrix of numbers), the update rule is simple: `weight -= learning_rate * gradient`. If a weight's gradient is positive, that weight made the loss worse - so we decrease it. If negative, we increase it. The learning rate controls how big each step is. Too small and training crawls. Too large and it overshoots, making the loss jump around instead of decreasing.

**Optimizers** are smarter versions of this update rule. SGD (Stochastic Gradient Descent) is exactly what we described above - subtract learning rate times gradient. It works, but it treats every parameter the same. Adam adapts the learning rate per parameter based on how the gradients have been behaving. Parameters with consistently large gradients get smaller steps; parameters with small, noisy gradients get larger steps. Adam is the default choice for most problems. SGD with momentum is sometimes preferred for large-scale training where memory matters.

**Epochs and Batches** control how we feed data to the training loop. An epoch is one complete pass through all training data (e.g we have run forward pass on ALL rows). A batch is a chunk of data processed together in one forward/backward pass. For Titanic (712 samples), we process everything at once. For ImageNet (1.2 million images), we process batches of 64 or 128 at a time. Batches exist because of memory constraints. Epochs exist because one pass through the data isn't enough - the network needs many passes to converge / find the “valley”.

## Terminology

| Term | What it means | Where we see it |
| --- | --- | --- |
| **Forward pass** | Data flowing through layers to produce predictions | Input tensor through linear layers, ReLU, sigmoid |
| **Tensor** | Multi-dimensional array (PyTorch's core data structure) | Input data, weights, gradients - everything is a tensor |
| **Weight matrix** | The learnable parameters in a linear layer | Each row is one neuron's recipe for combining inputs |
| **Bias** | An offset added after multiplication | Lets neurons activate even when inputs are zero |
| **ReLU** | Activation that zeros negatives, keeps positives | Between hidden layers - adds non-linearity |
| **Loss** | Single number measuring prediction error | Binary cross-entropy penalizes confident wrong answers |
| **Backpropagation** | Computing gradients by tracing backwards through operations | `loss.backward()` fills `.grad` on every parameter |
| **Gradient** | How much each weight contributed to the loss | Stored on parameters after backward pass |
| **Autograd** | PyTorch's automatic differentiation engine | Tracks operations, computes chain rule automatically |
| **Learning rate** | Step size for weight updates | Controls how much weights change per iteration |
| **SGD** | Stochastic Gradient Descent - basic optimizer | `weight -= lr * gradient` |
| **Adam** | Adaptive optimizer - adjusts LR per parameter | Default choice, converges faster than SGD on most problems |
| **Epoch** | One complete pass through all training data | Outer loop: repeat the full dataset multiple times |
| **Batch** | Chunk of data processed in one forward/backward pass | Inner loop: process 64 samples at a time |
| **Hyperparameter** | Setting you choose before training (not learned) | Learning rate, batch size, number of epochs, architecture |

## Connection to L4 and L6

**From L4:** We built the MLP architecture and trained it on Titanic. We introduced hidden layers, ReLU, sigmoid, and cross-entropy loss as concepts. L5 opens up every one of those concepts and shows what happens inside - the actual numbers, the actual matrix multiplications, the actual gradient values. The model code is the same; the understanding is completely different.

**To L6:** Now that you understand the training loop, L6 focuses on everything that happens before and after it. Data preprocessing (handling missing values, normalization, encoding categoricals), building proper data pipelines with DataLoaders, and evaluation (confusion matrix, precision, recall). L5 gives you the engine. L6 gives you the full car - from raw data to evaluated model.

## Resources - TWO VERY IMPORTANT NOTES FIRST:

1. **REMEMBER**! A lot of videos introduce neural networks as a way to classify images. I’ve always personally found that unintuitive - because they are also just as useful with standard tabular data. That means that for many tutorials, the input data is actually a matrix of pixel values, instead of values from columns you’d find in a standard database. And unfortunately, there isn’t much content on people teaching neural networks from the tabular data perspective that we’ll use in this courses notebook (we’ll do images with neural networks later). Think about it: how weird isn’t it, that each column in the matrix represents one pixel in an image? I’d rather start with data that is easy to interpret, and use a neural network for that, e.g the titanic data, or customer sales data!
2. Do not watch tutorials using tensorflow, unless you do it mainly to understand the concepts.
3. My lesson on this will probably be very important as I try to show more how it works

### Neural networks

A SMALL NOTE: My advice is that you try to use AI a lot to prompt yourself to understanding. I think the best way to learn is probably to create your own jupyter notebook where you ask the AI to take you though the entire process of training a neural network from scratch - as we’ll do in the L5 notebook “inside the neural network”.

- https://www.youtube.com/watch?v=XxZ0BibMTjw ←The only video I could find that uses tabular data for neural networks to teach it from the start, unfortunately it’s not the best
- StatQuest - Neural Networks Pt. 1: Inside the Black Box (20 min, builds from simple models to neural nets, clear and methodical, great complement to 3Blue1Brown): https://www.youtube.com/watch?v=CqOfi41LfDw
- 3Blue1Brown - But What is a Neural Network? (19 min, the best visual explanation of neural networks available, walks through layers, neurons, and activations with animations): https://www.youtube.com/watch?v=aircAruvnKk
- 3Blue1Brown - Gradient descent, how neural networks learn (21 min, visualizes the loss landscape and how gradient descent navigates it): https://www.youtube.com/watch?v=IHZwWFHWa-w
- StatQuest - Neural Networks Pt. 3: ReLU in Action (10 min, shows exactly what ReLU does to a neural network's output, with step-by-step examples): https://www.youtube.com/watch?v=68BZ5f7P94E
- StatQuest - Neural Networks Pt. 6: Cross Entropy (11 min, builds the intuition for why cross-entropy works as a loss function for classification): https://www.youtube.com/watch?v=6ArSys5qHAU
- Activation functions: https://www.youtube.com/watch?v=XxZ0BibMTjw <-- You mostly just have to look at ReLU and sigmoid.
- Activation functions: https://www.youtube.com/watch?v=s-V7gKrsels&t=12s <-- You mostly just have to look at ReLU and sigmoid.

### Quick videos

- StatQuest - Stochastic Gradient Descent, Clearly Explained (11 min, builds from basic gradient descent to SGD and mini-batches): https://www.youtube.com/watch?v=vMh0zPT0tLI
- Deeplizard - PyTorch Training Loop Explained (short, walks through the forward-backward-update pattern in code): https://www.youtube.com/watch?v=XAkN07mEzpo

### Practical sources

### Full tutorial series

- Patrick Loeber - PyTorch Tutorial series (comprehensive, covers tensors through training loops and beyond) - this is a great start with pytorch, but we’ll do more of the practical work in the next lesson: https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
- A tutorial that goes through the practical parts on how to use an eural network in pytorch for tabular data https://simulationbased.com/2021/02/03/a-deep-feedforward-network-in-pytorch-for-the-titanic-challenge/