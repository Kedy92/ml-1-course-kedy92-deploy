# L10: Images - MLP Practice (CIFAR-10)

In L09 we built the complete image pipeline from scratch on a messy bird dataset - data cleaning, resizing, augmentation, normalization, softmax, CrossEntropyLoss, and evaluation. We also saw the MLP's fundamental limitation: it has zero spatial awareness. The pixel shuffle experiment proved it. This lesson is about practicing everything from L09 on a new dataset and pushing the MLP as far as it can go. You pick the architecture, the hyperparameters, the regularization. The question isn't "can we make this work" - it's "how far can we push it before we hit the wall."

CIFAR-10 is cleaner than the bird dataset. All images are pre-sized at 32x32, pre-split into train and test, and evenly distributed across 10 classes. That means less data wrangling and more time experimenting with the model itself - which is exactly the point.

## Key Concepts

**Architecture experimentation.** In L04 and L05 we used simple MLPs without much thought about size. Now we experiment deliberately. How many neurons per layer (width)? How many layers (depth)? Which activation function? There's no formula - you try things, compare results, and build intuition for what matters.

**Hyperparameter tuning.** The learning rate is the single most important hyperparameter. Too low and training crawls. Too high and loss explodes. Batch size trades off gradient noise against speed. Epochs control how long you train - stop too early and you underfit, too late and you overfit. The discipline from L02 applies: change one thing, measure, compare.

**Regularization.** When the model memorizes training data instead of learning general patterns, you need regularization. Three tools:

- **Dropout** randomly zeros out neurons during training. This forces the network to build redundant representations instead of relying on any single pathway.
- **Weight decay** penalizes large weights by adding a small fraction of the weight magnitude to the loss. This keeps the model simpler.
- **Batch normalization** normalizes activations between layers. It stabilizes training and acts as a mild regularizer. We'll go much deeper on batch norm in L11.

**Reading loss curves.** This is one of the most important diagnostic skills in ML. Four patterns to recognize:

- Training loss drops, validation loss drops: learning is happening, keep going
- Training loss drops, validation loss rises: overfitting, add regularization or stop earlier
- Both losses plateau: model capacity is maxed out, or learning rate is too low
- Both losses stay high: underfitting, need more capacity or better features

You should be able to glance at a loss curve and immediately diagnose what's happening. This skill transfers to every model you'll ever train.

**The MLP ceiling.** After trying every trick - wider layers, deeper networks, dropout, weight decay, batch norm, tuned learning rates - the MLP will plateau somewhere around 50-55% on CIFAR-10 (random chance is 10%). That's not bad, but a simple CNN will blow past it. The architecture fundamentally can't capture spatial patterns, and no amount of hyperparameter tuning fixes a structural limitation.

## Terminology

| Term | What it means | Where we see it |
| --- | --- | --- |
| **Dropout** | Randomly zeroing neurons during training to prevent co-dependency | `nn.Dropout(0.3)` between layers |
| **Weight decay** | Penalizing large weights to keep the model simple | `optimizer = Adam(params, weight_decay=1e-4)` |
| **Batch normalization** | Normalizing activations between layers for stable training | `nn.BatchNorm1d()` after linear layers |
| **Learning rate scheduler** | Adjusting the learning rate during training | Start high, decay over time |
| **Regularization** | Techniques that prevent overfitting | Dropout, weight decay, augmentation, early stopping |
| **Underfitting** | Model too simple or undertrained, both losses are high | Need more capacity or longer training |
| **Overfitting** | Model memorized training data, validation loss rises | Need regularization or more data |
| **Loss curves** | Plots of training and validation loss over time | Primary diagnostic tool |

## Connection to L09 and L11

**From L09:** You built the pipeline once with guidance. Now you build it again independently on a different dataset. The data cleaning step is lighter (CIFAR-10 is clean), but every other piece of the pipeline - transforms, normalization, augmentation, model building, training, evaluation - carries over directly.

**To L11:** After pushing the MLP as far as it goes, you'll see it hit a hard ceiling. No amount of hyperparameter tuning fixes the MLP's structural inability to see spatial patterns. L11 introduces CNNs - an architecture designed from the ground up for images. Fewer parameters, higher accuracy, actual spatial awareness. Everything you've learned about training loops, loss curves, and evaluation still applies. The only thing that changes is the architecture.

## Resources

### Course video

TBA

### Documentation

- Google ML Crash Course - Overfitting (interactive explanation, useful for building intuition around loss curves): https://developers.google.com/machine-learning/crash-course/overfitting/overfitting
