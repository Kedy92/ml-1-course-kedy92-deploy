# L09: Images - The Complete Pipeline with MLP

In L06 we built a full tabular pipeline: load data, preprocess, train an MLP, evaluate. In L07 we saw that trees often beat neural nets on tabular data. Now we switch to a data type where neural networks truly shine - images. This lesson takes the same MLP architecture you already know and applies it to image classification. The twist: real images are messy. Unlike CIFAR-10 or Fashion-MNIST where everything is pre-cleaned and pre-sized, our bird dataset is web-scraped, with different resolutions, corrupted files, and class imbalance. Every pipeline step you'd skip with a toy dataset becomes necessary here.

The goal is to build the entire image classification pipeline from scratch, understand every piece, and then see exactly where the MLP hits its ceiling on images. That ceiling is the motivation for CNNs in L11.

## Key Concepts

**Images as data.** An image is just a grid of numbers. A grayscale image is one 2D grid (height x width). A color image is three grids stacked - one for red, one for green, one for blue. Each value ranges from 0 to 255. To feed an image into an MLP, we flatten it from a 2D grid into a 1D vector, the same way we fed a row of tabular features into the MLP in L05. A 128x128 color image becomes a vector of 128 * 128 * 3 = 49,152 numbers. This is already a lot of input features, and the images are small.

**Data cleaning.** Web-scraped data has problems that curated datasets hide from you. Some files are corrupted. Some aren't actually images. Some are too small to be useful. We verify every image loads before training, because a single bad file can crash a training loop hours in.

**Resizing strategies.** Neural networks need all images to be the same size. But images come in all shapes. We look at three strategies: squash (distort the aspect ratio to hit the target size), crop (cut to a square and lose edges), and resize-then-crop (resize slightly larger, then center crop). Each has tradeoffs - seeing the visual difference matters more than reading about it.

**transforms.Compose().** This is the image equivalent of a pandas preprocessing chain. You stack transforms in order - resize, augment, convert to tensor, normalize - and they run as a pipeline. Same concept as chaining `.fillna().apply().fit_transform()` in tabular, just for pixels.

**Per-channel normalization.** Just like we used StandardScaler on tabular features, we normalize each RGB channel to mean 0 and standard deviation 1. We compute the mean and std across the entire training set per channel. This helps training converge and is required if you ever use pretrained models.

**Data augmentation.** This is something that has no tabular equivalent. We randomly flip, rotate, crop, and adjust colors on training images so the model sees slightly different versions each epoch. This prevents memorization and effectively multiplies your training data for free. Critical rule: only augment training data, never validation.

**Multi-class classification.** In L04-L06 we did binary classification with sigmoid. Now we have 5 bird classes, so we need softmax - it takes raw scores (logits) and converts them into probabilities across all classes that sum to 1. The loss function changes too: CrossEntropyLoss replaces BCELoss. We trace through the math with real numbers to make it concrete.

**The MLP ceiling.** After training, we run the pixel shuffle experiment: randomly permute all pixel positions, retrain, and watch the accuracy barely change. The MLP treats every pixel as independent - it has zero spatial awareness. It can't tell that nearby pixels form edges or shapes. This is the fundamental limitation that CNNs solve.

## Terminology

| Term | What it means | Where we see it |
| --- | --- | --- |
| **RGB channels** | Three grids (red, green, blue) that make up a color image | Each pixel has 3 values |
| **Flattening** | Reshaping a 2D image grid into a 1D vector for the MLP | 128x128x3 becomes 49,152 inputs |
| **transforms.Compose()** | Pipeline of image preprocessing steps | Resize, augment, normalize in sequence |
| **Per-channel normalization** | Normalizing each color channel to mean 0, std 1 | Like StandardScaler per RGB channel |
| **Data augmentation** | Random transforms to create training variety | Flips, rotations, color jitter |
| **Softmax** | Converts raw scores to probabilities that sum to 1 | Multi-class output layer |
| **CrossEntropyLoss** | Loss function for multi-class classification | Combines log-softmax and NLL |
| **Top losses** | Images the model got most wrong | Debugging tool for finding data issues |
| **Pixel shuffle** | Randomly rearranging pixel positions | Proves MLP ignores spatial structure |
| **LR finder** | Sweep learning rates to find the best one | Plot loss vs learning rate, pick steepest descent |

## Connection to L08 and L10

**From L08:** L08 was project time to consolidate tabular ML. You've built MLPs, trained them, evaluated them, and compared them to trees. All of those skills carry forward - the training loop, loss curves, confusion matrices, the experimentation mindset. The difference now is the data type and the preprocessing it demands.

**To L10:** L10 is practice time. You'll apply this same pipeline to CIFAR-10 independently, experiment with architectures and hyperparameters, push the MLP as far as it can go, and see firsthand where it hits its ceiling. That ceiling sets up L11, where CNNs break through it.

## Resources

### Course video

TBA

### Videos

- Patrick Loeber - PyTorch Tutorial series (good reference for PyTorch patterns we use in the notebook): https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4

### Documentation

- PyTorch torchvision transforms: https://pytorch.org/vision/stable/transforms.html
- PyTorch Dataset & DataLoaders tutorial: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- PyTorch CrossEntropyLoss docs: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
