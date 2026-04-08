# L04: Classification + Neural Networks (Top-Down)

In Lesson 3, we built linear regression from scratch - predicting house prices with gradient descent, MSE loss, and a training loop. You saw the full cycle: forward pass, compute loss, backward pass, update weights. That loop is the backbone of all ML training, and it doesn't go away. What changes in L4 is the question we're asking.

Instead of "how much?" we're asking "which one?" The output isn't a number like $347,000 - it's a category. Survived or died. Spam or not spam. Benign or malignant. The model's job is to output a probability: "I'm 87% confident this passenger survived."

That shift from numbers to categories is the difference between regression and classification. And it turns out we only need two new ingredients to make the jump: a function that squashes any number into a probability (sigmoid), and a loss function designed for yes/no answers (cross-entropy). Everything else - gradient descent, the training loop, weight updates - stays the same.

Once we have logistic regression working, we'll hit its limit: it can only draw straight decision boundaries (e.g split data into groups using a straight line). Some problems need curves. That's where neural networks enter. We'll build a Multi-Layer Perceptron (MLP) in PyTorch, train it on the same data, and watch it outperform logistic regression. The MLP learns its own features - curved, nonlinear boundaries that capture patterns logistic regression misses.

The goal isn't to fully understand what's happening inside the network yet. That's L5. The goal here is to see it work and build intuition for why adding layers and activation functions changes what a model can learn.

## Key Concepts

**Sigmoid** - takes any number and squashes it to a value between 0 and 1. Large negative numbers become close to 0, large positive numbers become close to 1, and zero maps to exactly 0.5. This is the bridge between regression (any output) and classification (probability output). In L3, our model output could be any real number. Wrapping it in sigmoid gives us something we can interpret as "chance of survival."

**Logistic regression** - same linear model from L3 (z = wx + b), but we pass the output through sigmoid before making a prediction. If sigmoid(z) > 0.5, predict "survived." Otherwise, predict "died." It's still using gradient descent to learn the weights - the only change is the activation at the end and the loss function.

**Cross-entropy loss** - replaces MSE from L3. The intuition: it heavily penalizes confident wrong predictions. If the model says "99% chance this person survived" and they actually died, the loss spikes. If it says "51% survived" and gets it wrong, the penalty is much smaller. This is exactly what we want - push the model to be honest about its uncertainty.

**Decision boundary** - the line (or curve) where the model switches from predicting one class to the other. For logistic regression, this is always a straight line. You can engineer polynomial features to get curves, but that requires manual feature design. Neural networks learn curves automatically.

**The MLP (Multi-Layer Perceptron)** - logistic regression with hidden layers in between. Instead of going straight from input to output, the data passes through intermediate layers that learn to create useful feature combinations. Each layer transforms the data, making the next layer's job easier. The key insight: the network builds its own features instead of you engineering them. Don’t worry, we will spend a lot of time learning how this works, it will take multiple lessons - this is just the start.

**ReLU (Rectified Linear Unit)** - the activation function between hidden layers. It keeps positive values and zeros out negatives. Simple, but critical: without it, stacking linear layers just gives you another linear layer (matrix math collapses). ReLU introduces the non-linearity that lets the network learn curves. Sigmoid squashes to 0-1 at the output, ReLU bends the space between layers.

**Hidden layers** - the intermediate layers between input and output. Each neuron in a hidden layer computes a weighted sum of its inputs, applies ReLU, and passes the result forward. Think of them as the network building its own intermediate representations - features you didn't design but that turn out to be useful for the task. This is part of the “black box” that you might have heard of - a hidden layer will recognize patterns, but we don’t know what these patterns are deep down in neural networks, because gradient descent will optimize the numbers where it finds patterns.

## Terminology

| Term | What it means | Where we see it |
| --- | --- | --- |
| **Classification** | Predicting a category, not a number | Survived/died, benign/malignant |
| **Sigmoid** | Squashes any number to 0-1 range | Output layer of logistic regression |
| **Logistic regression** | Linear model + sigmoid for classification | Our first classifier on Titanic |
| **Cross-entropy loss** | Loss function that penalizes confident wrong answers | Replaces MSE from L3 |
| **Decision boundary** | Line/curve where model switches predictions | Visualized in 2D feature space |
| **MLP (Multi-Layer Perceptron)** | Neural network with hidden layers | Our upgrade from logistic regression |
| **Hidden layer** | Intermediate layer that learns feature combinations | 16 neurons between input and output |
| **ReLU** | Activation: keep positives, zero out negatives | Between hidden layers |
| **Activation function** | Non-linear function applied after a layer | Sigmoid (output), ReLU (hidden) |
| **Neuron** | One unit in a layer, computes weighted sum + activation | 16 neurons in our hidden layer |
| **Architecture** | The structure of a network (layers, sizes, activations) | Linear-ReLU-Linear-Sigmoid |
| **Parameters/weights** | Numbers the network learns during training | 97 total in our small MLP |

## From L3 to L4 and then to L5

**From L3:** Everything we built in the regression lesson carries over. The training loop (forward, loss, backward, update) is identical. Gradient descent still drives the learning. The difference is the output activation (sigmoid instead of raw output) and the loss function (cross-entropy instead of MSE). If L3 made sense, L4 is a small step - same engine, different task.

**To L5:** We've seen the MLP work and beat logistic regression, but we've treated it like a black box. L5 opens that box. We trace data through every layer - matrix multiplication, ReLU, another matrix multiplication, sigmoid. We see what hidden neurons actually compute, how loss.backward() calculates gradients, and how optimizers update weights. L4 gives you the "what" and the "why." L5 gives you the "how."

## Resources

### Before the lesson

### Classification

- A good intro video https://www.youtube.com/watch?v=3bvM3NyMiE0
- You can try to read their section on logistic regression, but you don’t have to spend all that much time on the classification section as I personally believe we’ll deal more with ta as we get into neural networks https://developers.google.com/machine-learning/crash-course/logistic-regression - logistic regression is one way of doing classification, but don’t try to memorize everything - we don’t actually use logistic regression barely at all, what’s better to spend time on is understanding things like the sigmoid function, and once again realizing that we’re generating random weights and try to optimize them using gradient descent
- StatQuest - Logistic Regression (15 min, clear walkthrough of sigmoid, log-odds, and maximum likelihood - the same model we build from scratch in the notebook): https://www.youtube.com/watch?v=yIYKR4sgzI8.

### A slight intro to neural networks - don’t focus much on this, but feel free to glance at it.

Activation functions: https://www.youtube.com/watch?v=s-V7gKrsels

Both the sigmoid function and ReLU are activation functions - where mathematical functions that can take an input and give an output. The two most important are sigmoid and ReLU. ReLU is something absolutely critical to make sure neural networks actually work! They are part of the magic, if you will. And they are far more easy than you would think. 

If you go through the notebook, you’ll understand that the sigmoid function “squashes” values to be something between 0 and 1 - cool! That means that if you had a negative value, it might end up something like 0.1. ReLU on the other hand just says: if I ever get a negative value, I’m just going to make it 0! That’s it. It turns negative values to 0, and positive numbers stay exactly what they are, no change whatsoever. In the next lecture, we’ll talk even more about ReLU - but for now, just understand that being able to take a number and make sure its either 0 or positive is something that benefits neural networks, because it adds NON LINEARITY! 

- You can start watching this series a little bit, but it’s better to spend more time on it for the next lesson https://www.youtube.com/watch?v=kY14KfZQ1TI&list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1&index=1

### Documentation and tutorials

- Google ML Crash Course - Logistic Regression (interactive text, decent starting point for the math): https://developers.google.com/machine-learning/crash-course/logistic-regression
- Medium - Train a Neural Network in PyTorch (beginner walkthrough, covers the same ground as our notebook): https://medium.com/@sahin.samia/train-a-neural-network-in-pytorch-a-complete-beginners-walkthrough-3897d18d6078
- PyTorch - Building Models tutorial (official docs, reference for nn.Module and nn.Sequential): https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html