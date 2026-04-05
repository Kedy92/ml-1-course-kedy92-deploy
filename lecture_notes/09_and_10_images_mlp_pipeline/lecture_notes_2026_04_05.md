# L09 & L10: MLPs on Images

In L4-L6, we built neural networks on tabular data: a handful of curated features per row (4 for Titanic, ~40 for bank marketing), each with a clear meaning. The MLP learned weight combinations to classify passengers or predict outcomes. In L7, we saw that tree models often beat MLPs on this kind of data.

Now I figure we’ll switch to images - the domain where neural networks dominate. The central question of this lesson: **can the same MLP architecture that worked on Titanic work on images?** What changes, what stays the same, and where does it break?

The answer, as you'll discover: the MLP works - kind of. It learns something real. But it hits a hard ceiling that no amount of tuning can fix. Understanding why is what motivates CNNs in L11. We’ll spend 2 lessons learning, and a lot of it will carry over to CNN’s, and we’ll solidify our knowledge about MLP’s. 

## Images as Data

A color image is three 2D grids stacked: Red, Green, Blue. Each pixel has 3 values (0-255). A 128x128 color image = 128 x 128 x 3 = **49,152 numbers**. Compare that to a Titanic passenger's 4 features.

To feed an image into an MLP, we **flatten** that 3D block into a single 1D vector - the same way a tabular row is a 1D vector of feature values. And here's the problem: flattening destroys spatial information. Pixel (0,0) and pixel (127,127) become just two numbers in a long list. The MLP has no idea they're at opposite corners. We'll prove this matters with an experiment at the end of the notebook.

PyTorch uses **NCHW** tensor ordering: batch size, channels, height, width. A batch of 32 color images at 128x128 is shaped `(32, 3, 128, 128)`.

## From Binary to Multi-class

In L4/L5, every model answered a yes/no question: one output neuron, sigmoid activation, BCELoss. Now we have 5 bird classes, which means we need a new output strategy.

**Softmax** replaces sigmoid. Instead of one neuron outputting a probability, we have 5 output neurons (one per class), and softmax converts their raw values (logits) into probabilities that sum to 1. The classes compete - if confidence in "owl" goes up, confidence in everything else goes down. To make a prediction, we pick the class with the highest probability (`argmax`).

**CrossEntropyLoss** replaces BCELoss. It only cares about the probability assigned to the correct class: `Loss = -log(p_correct)`. High confidence in the right answer = low loss. Confident wrong answer = very high loss. The -log curve is gentle near p=1 but shoots toward infinity near p=0, which is what drives the model to avoid confident mistakes.

**PyTorch gotcha:** `nn.CrossEntropyLoss` does softmax internally. Your model's last layer should output raw logits (plain `nn.Linear`), not softmax probabilities. Applying softmax before CrossEntropyLoss is double-softmax and will silently produce wrong results.

## The Image Pipeline

The pipeline mirrors L6. Same structure, same logic, different tools. In L6 we went from raw CSV to evaluated model. Here we go from raw photos to evaluated model.

### Loading and cleaning

In L6, loading was `pd.read_csv()` and cleaning was handling missing values and outliers. For images, we load from a folder structure where each subfolder name is the class label. Cleaning means verifying every image file actually opens - web-scraped data has corrupted files, truncated downloads, and files that aren't real images. One bad file in your DataLoader crashes training at a random batch.

### Preprocessing

In L6, we used StandardScaler to normalize tabular features. For images, preprocessing has more steps: resize all images to the same dimensions (neural networks need fixed input sizes), convert to tensors, and normalize each RGB channel to mean 0 and std 1. The normalization is the same idea as StandardScaler, just applied per color channel instead of per column.

All of these steps are chained together with `transforms.Compose()` - the image equivalent of a preprocessing pipeline.

### Data augmentation

This is the one step with no tabular equivalent. You can't meaningfully flip a Titanic passenger, but a horizontally flipped parrot is still a parrot. Augmentation randomly transforms training images each epoch (flips, rotations, color changes) so the model sees slightly different versions every time. With only ~200 images per class, this is essential - without it, the model memorizes exact pixel patterns.

Important: only augment training data. Validation images get clean, deterministic transforms so your evaluation is consistent.

### Training

The training loop is the same one from L6, but we add three new tools:

- **LR finder** - instead of guessing a learning rate, sweep from tiny to large and plot loss. Pick where loss drops fastest. With 25 million parameters, the rate that worked for 91 parameters won't work here.
- **Learning rate scheduler** (`ReduceLROnPlateau`) - halves the learning rate when validation loss stops improving. Big steps early to cover ground, smaller steps later to settle in precisely.
- **Early stopping** - save the model when validation loss improves, stop after `patience` epochs with no improvement, restore the best model (not the final one).

### Evaluation

Same tools as L6 (confusion matrix, per-class accuracy, loss curves), plus **top losses analysis** - sorting validation images by how wrong the model was and looking at the worst ones. This tells you whether failures are bad data, genuinely hard cases, or systematic weaknesses. It's one of the most useful debugging tools for image models.

## What the MLP Actually Learns

The MLP sees images as flat vectors of numbers. It has no concept of edges, shapes, or objects. What it learns instead is **color and texture statistics**: "lots of blue pixels probably means sky, which correlates with eagle." This works up to a point, but it can't distinguish birds with similar color palettes.

The notebook proves this with the **pixel shuffle experiment**: randomly rearrange all pixel positions and retrain. If the MLP truly ignores spatial structure, shuffling shouldn't hurt. Result: the MLP performs similarly on shuffled images. It was never using spatial information.

The other problem is **parameter explosion**: the first layer alone needs 25 million parameters for 128x128 images. At 224x224, that jumps to 77 million - and a CNN like ResNet-18 achieves better accuracy with 11.7 million total. The MLP learns separate weights for every pixel position. It can't reuse what it learns about patterns in one part of the image anywhere else.

These two limitations - no spatial awareness, parameter explosion - are exactly what CNNs solve in L11.

## Regularization and Training Strategies (L10 focus)

Once the pipeline is built and the baseline model trains, the question becomes: how do we push accuracy higher without changing the architecture fundamentally? This is what L10 explores through 5 model versions, each adding one technique.

**The overfitting problem**: with 25 million parameters and only a few thousand training images, the MLP will memorize the training set if you let it. Training accuracy climbs toward 100% while validation accuracy plateaus. Every technique below fights this in a different way.

### Regularization

**Dropout** randomly zeros out a percentage of neurons each training step (e.g. 30%). This forces the network to build redundant representations - no single neuron can become a bottleneck. At evaluation time, all neurons are active. Think of it as training an ensemble of smaller networks that share weights. Typical values: 0.2-0.5.

**Weight decay** (L2 regularization) adds a penalty to the loss for large weights: the optimizer prefers smaller, more distributed weight values over a few large ones. You set it directly in the optimizer: `SGD(weight_decay=1e-4)`. It keeps the model simpler without changing the architecture.

**BatchNorm** normalizes activations between layers, keeping them centered and preventing them from drifting or collapsing. It's technically not regularization - it's a training stability technique - but it has a mild regularizing effect because batch statistics add noise. L11 goes much deeper into why BatchNorm matters.

### Optimizers

In L5/L6, we used Adam and it worked. For images, **SGD with momentum** is worth trying. Momentum uses a running average of past gradients for smoother updates - instead of reacting to each batch independently, the optimizer builds up "velocity" in consistent directions. The tradeoff: SGD needs more tuning (learning rate, momentum value) but often finds better solutions than Adam on image tasks.

### Learning rate strategies

A fixed learning rate is rarely optimal for the full training run. Two schedulers appear in L10:

- **ReduceLROnPlateau** - watches validation performance and halves the learning rate when it stops improving. Simple and reactive.
- **Cosine annealing** - smoothly decays the learning rate following a cosine curve from start to finish. No decisions needed - just set the total epochs.

Both serve the same intuition: big steps early to cover ground, smaller steps later to settle in precisely.

### Activation functions

**LeakyReLU** is a variant of ReLU that allows a small negative slope (e.g. 0.01x) instead of zeroing out all negatives. This prevents "dead neurons" - neurons whose output is always negative, meaning their gradient is always zero and they never update. In practice the difference is small, but it's worth knowing.

### Weight initialization

**Kaiming initialization** scales initial weights based on the number of inputs to each layer. Without it, activations can explode or vanish at the start of training, especially in wider networks. PyTorch's defaults work for small models, but explicitly applying Kaiming helps when you scale up. L11 explores this further.

### The experimentation discipline

Change one thing at a time. Train, compare to your baseline, decide whether to keep the change. If you change five things at once, you won't know which one helped. The L10 notebook walks through this progression: baseline -> dropout -> BatchNorm -> combined optimizations -> maximum effort.

## Terminology

| Term | What it means |
| --- | --- |
| **RGB channels** | Three grids (red, green, blue) making a color image, each pixel 0-255 |
| **Flattening** | Reshaping a 3D image into a 1D vector (128x128x3 = 49,152 inputs) |
| **NCHW** | PyTorch tensor layout: batch, channels, height, width |
| **Softmax** | Converts logits to probabilities summing to 1 (replaces sigmoid) |
| **CrossEntropyLoss** | Multi-class loss function, expects raw logits (replaces BCELoss) |
| **Data augmentation** | Random image transforms (flip, rotate, jitter) to prevent memorization |
| **Per-channel normalization** | Normalizing each RGB channel to mean 0, std 1 |
| **BatchNorm / Dropout** | Regularization techniques: stabilize activations / zero random neurons |
| **LR finder** | Sweep learning rates to find the optimal one |
| **Early stopping** | Stop training when validation loss stops improving |
| **Top losses** | Images the model got most wrong - debugging tool |

## Notebooks

- **L9**: `lessons/neural_networks_images/09_and_10/09/testing/09_mlp_image_classification.ipynb` - guided walkthrough, a custom bird dataset which I built
- **L10**: `lessons/neural_networks_images/09_and_10/10/testing/10_mlp_image_classification_project.ipynb` - self-directed practice, CIFAR-10

## Where This Fits

**From L7/L8:** L7 showed that trees often beat MLPs on tabular data. All the skills you built - training loops, loss curves, confusion matrices - carry forward. The data type changes, the pipeline structure stays the same.

**L9 -> L10:** L9 is the guided walkthrough, L10 is practice. You apply the pipeline independently to CIFAR-10 and experiment with regularization strategies. The MLP will plateau around 55% no matter what you try - that ceiling is the point.

**To L11 (CNNs):** The MLP's two failures - no spatial awareness, parameter explosion - are exactly what CNNs solve. Instead of connecting every pixel to every neuron, CNNs use small filters that slide across the image: spatial structure is preserved, and a single 3x3 filter (27 parameters) replaces millions of connections. L11 builds CNNs from scratch on MNIST, then CIFAR-10.

**To L12 (Fine-tuning):** In practice, you almost never train a CNN from scratch. You've already done this in L1/L2 and Assignment 1 - taking a pretrained model and adapting it. L12 goes deeper into how and why transfer learning works, and how to do it properly in PyTorch.

**The bigger picture:** Classification ("what's in this image?") is just the starting point. Once you have a CNN that understands images, you can extend it to detection ("where is each object?"), segmentation ("which pixels belong to what?"), and real-time systems like YOLO. L9-L10 establish the foundation that everything else builds on.

## Resources

### Before the lesson

This is the first lesson where most external tutorials will actually match what we're doing - because the standard way to teach neural networks IS with images. The tutorials that felt mismatched in L4-L5 (because they used images while we used tabular data) now become directly relevant.

## MUST WATCH - Now that we're working with images, I think Patrick Loebers course on pytorch is a great complement to the course, as he includes a lot of theoretical videos (e.g softmax, cross entropy)

- Patrick Loeber - PyTorch Tutorial series (good reference for PyTorch patterns used in the notebook): https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
- 3Blue1Brown - But What is a Neural Network? (19 min, visual explanation of layers and activations - if you haven't watched this yet, now is the time): https://www.youtube.com/watch?v=aircAruvnKk

### Videos

**Softmax & Cross entropy loss (it's up to you if you care about the math)**

- A good summary of softmax (remember, its another way of turning something into a probability, but this time since we might be doing multi classification we'll need to turn many outputs into probabilities) https://www.youtube.com/watch?v=oJU6-qW6xZU
- Softmax & argmax, more in detail, if you care (dont worry about argmax, just a sidenote in the video) and https://www.youtube.com/watch?v=KpKog-L9veg
- Cross entropy loss, a good walkthrough: https://www.youtube.com/watch?v=7q7E91pHoW4
- Cross entropy loss (remember, its just a way to calculate the loss when we have multiple probabilities): https://www.youtube.com/watch?v=6ArSys5qHAU
- Data Augmentation explained (short, covers why and when to augment): https://www.youtube.com/watch?v=JI8saFjK84o
- This video is also great and more practical on data augmentation: https://www.youtube.com/watch?v=qLIosWyrh9Q&list=PLTKMiZHVd_2KJtIXOW0zFhFfBaJJilH51&index=74

### Regularization

Sebastian Raschka has some good videos on regularization, in fact he has a great playlist on ML with various topics that are great, I would watch all the videos in L10-L11 for e.g regularization, but you'll find many sections that are great:
Overview:
https://www.youtube.com/watch?v=Va4K-wYh_p8&list=PLTKMiZHVd_2KJtIXOW0zFhFfBaJJilH51&index=72
E.g for early stopping:
https://www.youtube.com/watch?v=YA1OdkiHJBY&list=PLTKMiZHVd_2KJtIXOW0zFhFfBaJJilH51&index=75

### Documentation

- PyTorch torchvision transforms (reference for all available image transforms): https://pytorch.org/vision/stable/transforms.html
- PyTorch Dataset & DataLoaders tutorial: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- PyTorch CrossEntropyLoss docs: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- PyTorch ImageFolder docs (the built-in dataset for folder-structured image data): https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html