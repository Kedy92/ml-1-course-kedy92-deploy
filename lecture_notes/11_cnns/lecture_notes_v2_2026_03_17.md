# L11: CNNs - Concept and Architecture

In L09 and L10 we pushed the MLP as far as it could go on images. We tried wider layers, deeper networks, dropout, weight decay, batch norm, tuned learning rates - and hit a ceiling. The pixel shuffle experiment proved why: the MLP treats every pixel independently. It has zero spatial awareness. A cat's ear next to its eye means nothing more than a random pixel somewhere else in the image. CNNs fix this. They're an architecture designed specifically for spatial data, and they solve both of the MLP's core problems at once: the parameter explosion (flattening a 224x224x3 image gives 150,528 inputs) and the lack of spatial awareness.

This lesson builds a CNN from scratch in pure PyTorch. No fastai, no pretrained models - every piece is visible. We start with hand-crafted edge detection kernels so you can see exactly what a convolution does, then let the network learn its own kernels through training. By the end, we train a CNN on MNIST that crushes the MLP on the same task with fewer parameters.

## Key Concepts

**The convolution operation.** A kernel (also called a filter) is a small grid of weights - typically 3x3. It slides across the image one position at a time, computing a dot product at each position: multiply element-wise, sum the results. The output is a feature map - a new 2D grid that highlights where the kernel's pattern was found. We start by hand-crafting edge detection kernels (top edges, left edges) so you can see the operation produce meaningful results before any learning happens.

**Feature maps.** Each kernel produces one feature map. A top-edge kernel produces a map that lights up wherever there are top edges. A diagonal-edge kernel lights up on diagonals. In early layers of a trained CNN, kernels detect edges, gradients, and textures. In deeper layers, they detect parts, shapes, and eventually whole objects. Nobody programs these - the network learns them through gradient descent, same as always.

**Channels and depth.** A grayscale image has 1 channel. A color image has 3 (RGB). When a convolution operates on a 3-channel input, the kernel is actually a 3x3x3 volume - one 3x3 slice per input channel. The slices are convolved separately and summed into one output. Each conv layer can have multiple output channels (one kernel per output channel), so depth grows through the network. The pattern: spatial dimensions shrink while channel count grows, funneling information from raw pixels to abstract features.

**Stride.** By default the kernel moves one pixel at a time (stride 1), producing an output the same size as the input (with padding). Stride 2 means the kernel skips every other position, halving the spatial dimensions. This is the modern replacement for max pooling - simpler, and it works just as well. The architecture pattern is: each stride-2 layer halves the spatial size and doubles the channels.

**Padding.** When a 3x3 kernel slides across a 28x28 image, the output is 26x26 - you lose one pixel on each side. Adding 1 pixel of padding around the image preserves the spatial dimensions. For a kernel of size `ks`, the padding needed to preserve size is `ks // 2`.

**Batch normalization.** This is the training stability tool that makes deep CNNs work. Without it, activations drift toward zero as they pass through layers - neurons "die" and stop learning. Batch norm normalizes activations to mean 0 and std 1 at each layer, then learns a scale and shift so the network can undo the normalization if it's not helpful. We spend significant time in the notebook diagnosing activation collapse without batch norm, then watching the fix work. The standard block becomes Conv - BatchNorm - ReLU.

**1cycle learning rate.** Instead of a fixed learning rate, 1cycle starts low, ramps up to a peak, then decays back down. The warmup phase lets the network find a reasonable region of the loss landscape before taking big steps. The decay phase fine-tunes. PyTorch's `OneCycleLR` uses cosine annealing for the decay, which works slightly better in practice than linear decay.

**CNN vs MLP.** On the same dataset, the CNN achieves higher accuracy with fewer parameters. The MLP needs one weight per pixel per neuron - parameter count explodes with image size. The CNN reuses the same small kernel across the entire image - a 3x3 kernel has 9 weights regardless of whether the image is 28x28 or 1024x1024. This weight sharing is what gives CNNs their efficiency and spatial awareness.

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

## Connection to L10 and L12

**From L10:** You pushed the MLP to its ceiling on images. You know the training loop, loss curves, evaluation tools, and regularization techniques. All of that carries forward - the CNN changes the architecture, not the process. The training loop is the same. The evaluation is the same. Loss curves still tell you the same stories.

**To L12:** Now that we understand how CNNs work from scratch, L12 shows the practical shortcut: take a CNN that someone already trained on millions of images (ResNet, EfficientNet) and fine-tune it on your data. We also introduce object detection with YOLO and deploy a model as a FastAPI endpoint. L11 gives you the foundation to understand what those pretrained models are actually doing under the hood.

## Resources

### Course video

TBA

### Videos

- CNN intro (clear visual walkthrough of convolutions and pooling): https://www.youtube.com/watch?v=pj9-rr1wDhM
- Alternative CNN intro (different angle, good for reinforcement): https://www.youtube.com/watch?v=HGwBXDKFk9I
- Practical convolutions in Excel (from ~44 min, Jeremy Howard walks through convolution arithmetic in a spreadsheet): https://www.youtube.com/watch?v=htiNBPxcXgo

### Documentation and papers

- Guide to Convolution Arithmetic (paper, visual explanations of padding, stride, and output sizes): https://arxiv.org/abs/1603.07285
- Zeiler and Fergus - Visualizing what CNNs learn (paper, the feature visualization images referenced in the notebook): https://arxiv.org/abs/1311.2901
- Batch Normalization (original paper by Ioffe and Szegedy): https://arxiv.org/abs/1502.03167
- PyTorch nn.Conv2d docs: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- PyTorch OneCycleLR docs: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
