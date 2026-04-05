# L11: CNNs - Concept and Architecture

In L9 and L10, we pushed the MLP as far as it could go on images. We tried wider layers, deeper networks, dropout, weight decay, batch norm, tuned learning rates - and hit a ceiling. The pixel shuffle experiment proved why: the MLP treats every pixel independently. It has zero spatial awareness. A cat's ear next to its eye means nothing more than a random pixel somewhere else in the image. CNNs fix this. They're an architecture designed specifically for spatial data, and they solve both of the MLP's core problems at once: the parameter explosion (flattening a 224x224x3 image gives 150,528 inputs) and the lack of spatial awareness.

This lesson builds a CNN from scratch in pure PyTorch. No fastai, no pretrained models - every piece is visible. We start with hand-crafted edge detection kernels so you can see exactly what a convolution does, then let the network learn its own kernels through training. By the end, we train a CNN on MNIST that crushes the MLP on the same task with fewer parameters.

## The Convolution Operation

A kernel (also called a filter) is a small grid of weights - typically 3x3. It slides across the image one position at a time, computing a dot product at each position: multiply element-wise, sum the results. The output is a feature map - a new 2D grid that highlights where the kernel's pattern was found.

We start with hand-crafted kernels (top edges, left edges, diagonals) so you can see the operation produce meaningful results before any learning happens. A top-edge kernel returns high values where there's a dark-to-light transition from top to bottom, and negative values for the opposite. You'll compute these by hand, then apply them across a full MNIST digit.

In practice, nobody designs these kernels by hand. The network learns whatever kernel values minimize the loss - same gradient descent as always.

## Feature Maps and Channels

Each kernel produces one feature map. A top-edge kernel produces a map that lights up wherever there are top edges. A diagonal-edge kernel lights up on diagonals. Apply 4 kernels and you get 4 feature maps.

In early layers of a trained CNN, kernels detect edges, gradients, and textures. In deeper layers, they detect parts, shapes, and eventually whole objects. Nobody programs these - the network learns them through gradient descent.

A grayscale image has 1 channel. A color image has 3 (RGB). When a convolution operates on a 3-channel input, the kernel is actually a 3x3x3 volume - one 3x3 slice per input channel. The slices are convolved separately and summed into one output. Each conv layer can have multiple output channels (one kernel per output channel), so depth grows through the network. The pattern: spatial dimensions shrink while channel count grows, funneling information from raw pixels to abstract features.

## Stride and Padding

**Padding.** When a 3x3 kernel slides across a 28x28 image, the output is 26x26 - you lose one pixel on each side. Adding 1 pixel of padding (zeros) around the image preserves the spatial dimensions. For a kernel of size `ks`, the padding needed to preserve size is `ks // 2`.

**Stride.** By default the kernel moves one pixel at a time (stride 1). Stride 2 means the kernel skips every other position, halving the spatial dimensions. This is the modern replacement for max pooling - simpler, and it works just as well. The architecture pattern is: each stride-2 layer halves the spatial size and doubles the channels.

## Building the CNN

With convolutions understood, we replace the MLP's linear layers with conv layers. The key difference: a linear layer needs a weight per pixel per neuron (parameter explosion). A conv layer reuses the same small kernel across the entire image - a 3x3 kernel has 9 weights regardless of whether the image is 28x28 or 1024x1024. This weight sharing is what gives CNNs their efficiency and spatial awareness.

The notebook builds a simple CNN for 3s vs 7s classification: four stride-2 conv layers that progressively shrink the spatial dimensions (28 -> 14 -> 7 -> 4 -> 2) while growing channels (1 -> 4 -> 8 -> 16 -> 2). The final 1x1 spatial output gets flattened to a class prediction.

## Receptive Fields

The receptive field is the area of the input image that affects one output value. A single stride-2 conv with a 3x3 kernel has a receptive field of 3x3. Stack two and it grows to 7x7. With enough layers, each output value "sees" the entire input image. This is how CNNs build from local patterns (edges) to global understanding (shapes, objects) - each layer combines the previous layer's features over a wider area.

## Color Images

Grayscale images have 1 channel. Color images have 3 (RGB). The notebook shows how convolution extends to color: each kernel becomes a 3D volume (one slice per channel), the per-channel results are summed, and the output is still a single feature map. Visualization of learned first-layer kernels (from Zeiler & Fergus) shows they naturally learn color-sensitive edge detectors - no one told the network about color.

## Improving Training Stability

The notebook scales from 2-class (3s vs 7s) to 10-class (all digits) and the simple CNN fails. Training barely improves over random. The diagnosis uses activation statistics (mean, std, percentage near zero) tracked per layer across training batches. The pattern: activations collapse toward zero in deeper layers, neurons "die" and stop contributing.

### Batch size experiment

Larger batches (64 -> 512) give smoother gradients but fewer updates per epoch. Doesn't fix activation collapse.

### 1cycle learning rate

Instead of a fixed learning rate, 1cycle starts low, ramps up to a peak, then decays back down with cosine annealing. The warmup phase lets the network find a reasonable region before taking big steps. The decay phase fine-tunes. PyTorch's `OneCycleLR` implements this. It helps, but doesn't fix the root cause.

### Batch normalization

This is the training stability tool that makes deep CNNs work. Without it, activations drift toward zero as they pass through layers - neurons "die" and stop learning. Batch norm normalizes activations to mean 0 and std 1 at each layer, then learns a scale and shift so the network can undo the normalization if it's not helpful.

The notebook diagnoses this in detail: a simulation showing distribution collapse without batchnorm vs. healthy distributions with it, then real MNIST activations confirming the same pattern. After adding batchnorm, the same model jumps from barely learning to 98%+ accuracy in a single epoch.

The standard block becomes **Conv -> BatchNorm -> ReLU**.

## CNN vs MLP

On the same dataset, the CNN achieves higher accuracy with fewer parameters. The MLP needs one weight per pixel per neuron - parameter count explodes with image size. The CNN reuses the same small kernel across the entire image. This weight sharing is what gives CNNs their efficiency and spatial awareness. The universal approximation theorem says a fully connected network *can* represent anything - but in practice, the right architecture makes learning dramatically easier.

## Terminology

| Term | What it means | Where we see it |
| --- | --- | --- |
| **Kernel (filter)** | Small grid of learnable weights (e.g. 3x3) | Slides across the image, detecting patterns |
| **Feature map** | Output of applying one kernel to an image | Highlights where the pattern was found |
| **Channels** | Depth dimension of the image or feature maps | RGB = 3 channels, conv layers add more |
| **Stride** | How many pixels the kernel moves per step | Stride 2 halves spatial dimensions |
| **Padding** | Extra pixels added around image edges | Preserves spatial dimensions after convolution |
| **Batch normalization** | Normalizes activations between layers | Prevents activation collapse, stabilizes training |
| **1cycle LR** | Learning rate schedule: warmup then decay | `OneCycleLR` in PyTorch |
| **Activation collapse** | Neurons drift to zero and stop learning | Diagnosed with activation statistics |
| **Receptive field** | Area of input that affects one output value | Grows with depth - deeper layers "see" more |
| **NCHW** | Tensor layout: batch, channels, height, width | Standard PyTorch image tensor format |
| **Weight sharing** | Same kernel weights applied across all positions | Why CNNs have fewer parameters than MLPs |

## Notebooks

- **L11**: `lessons/neural_networks_images/11_cnns/testing/L11_cnns.ipynb` - builds CNNs from scratch, hand-crafted kernels through batchnorm
- **project_oxford_pets**: `lessons/neural_networks_images/11_cnns/homework/modified/project_oxford_pets.ipynb` - example project I did based on the starting instructions
- **cnn_project_starting_instructions**: `lessons/neural_networks_images/11_cnns/homework/modified/cnn_project_starting_instructions.ipynb` - choose your own dataset and practice (HOMEWORK)

## Where This Fits

**From L9/L10:** You pushed the MLP to its ceiling on images. You know the training loop, loss curves, evaluation tools, and regularization techniques (dropout, weight decay, BatchNorm, LR scheduling). All of that carries forward - the CNN changes the architecture, not the process. The training loop is the same. The evaluation is the same. Loss curves still tell the same stories. The difference: CNNs solve the two MLP failures you saw firsthand - no spatial awareness and parameter explosion.

**To L12 (Fine-tuning and YOLO):** Now that you understand how CNNs work from scratch, L12 shows the practical shortcut: take a CNN that someone already trained on millions of images (ResNet, EfficientNet) and fine-tune it on your data. You've already done this in L1/L2 and Assignment 1 with fastai - L12 goes deeper into how and why transfer learning works, and how to do it properly in PyTorch. We also introduce object detection with YOLO and deploy a model as a FastAPI endpoint.

**The bigger picture:** Classification ("what's in this image?") is the starting point. Once you have a CNN that understands images, you can extend it to detection ("where is each object?"), segmentation ("which pixels belong to what?"), and real-time systems like YOLO. L11 gives you the foundation to understand what those pretrained models are actually doing under the hood.

## Resources

### Notebooks:

- l11_cnns - A notebook worth going through once you've watched the videos
- project_oxford_pets - more focus on practical skills, an example notebook I did based on the cnn_project_starting_instructions
- cnn_project_starting_instructions -> A starting point to choose your own dataset and do some practice (HOMEWORK)

### Videos
- Watch all of the videos on CNN's from this playlist: https://www.youtube.com/playlist?list=PLTKMiZHVd_2KJtIXOW0zFhFfBaJJilH51 (L13 videos, e.g L13.1, L13.2, L13.3)
First video in the section: https://www.youtube.com/watch?v=i-Ngb6tn_KM&list=PLTKMiZHVd_2KJtIXOW0zFhFfBaJJilH51&index=98
- This playlist is also great starting from his videos on "what are convulition layers" https://www.youtube.com/playlist?list=PLN8j_qfCJpNhhY26TQpXC5VeK-_q3YLPa

- CNN intro (clear visual walkthrough of convolutions and pooling): https://www.youtube.com/watch?v=pj9-rr1wDhM
- Alternative CNN intro (different angle, good for reinforcement): https://www.youtube.com/watch?v=HGwBXDKFk9I
- Practical convolutions in Excel (from ~44 min, Jeremy Howard walks through convolution arithmetic in a spreadsheet): https://www.youtube.com/watch?v=htiNBPxcXgo

### Documentation and papers

- Guide to Convolution Arithmetic (paper, visual explanations of padding, stride, and output sizes): https://arxiv.org/abs/1603.07285
- Zeiler and Fergus - Visualizing what CNNs learn (paper, the feature visualization images referenced in the notebook): https://arxiv.org/abs/1311.2901
- Batch Normalization (original paper by Ioffe and Szegedy): https://arxiv.org/abs/1502.03167
- PyTorch nn.Conv2d docs: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- PyTorch OneCycleLR docs: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
