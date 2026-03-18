# L02: Your First Model - The ML Process

In Lesson 1, we saw what ML can do - demos across tabular data, images, and text. We introduced the core vocabulary (features, labels, training, inference, loss, gradient descent) and saw models work without understanding how. Just a reminder: you’re not really expected to understand all of those words yet, that’s what we’re going to build on today, and the upcoming lessons will keep repeating a lot of these things - after all, we’ll keep doing the same thing over and over again, many times.

Now we're going to actually build something. This lesson walks through the full ML process from start to finish: loading data, preparing it, training a model, evaluating results, iterating, and saving the model for deployment. By the end, you'll have a working image classifier that can recognize 37 breeds of cats and dogs.

The purpose of this lesson is really to showcase a real process and pipeline when training a model. Along the way, we introduce a lot of the core concepts of machine learning - though some details won't be fully revealed until L3 and L4.

## The ML Process

Every ML project follows the same loop. This is the backbone of the entire lesson:

1. **Understand** your data - what do you have? What are you predicting?
2. **Prepare** the data - clean it, split it, set up the pipeline
3. **Train** the model - pick an architecture, fit it to the data
4. **Evaluate** - how good is it really? Where does it fail?
5. **Iterate** - try different settings, improve
6. **Ship** - save the model, deploy it

We did steps 1-4 briefly in the Titanic demo. Now we're doing all six properly, with real evaluation tools and a model you can actually deploy.

## What is PyTorch?

Before we talk about the tool we'll use today (fastai), let's talk about the tool we'll use for the rest of the course: PyTorch.

PyTorch is the dominant ML framework used in research and industry. When you read ML papers, when you see production models at Meta, Tesla, or OpenAI - they're usually PyTorch. It was created by Meta (Facebook) and is now the standard. TensorFlow used to compete, but PyTorch has essentially won.

At its core, PyTorch does two things:

1. **Tensor operations** - like numpy but with GPU support, so matrix math runs 10-100x faster (we’ll talk tensors during the course, as they are the core of pytorch)
2. **Automatic differentiation (autograd)** - it can automatically compute gradients (derivatives), which is what makes gradient descent work without manual math. We’ll also talk more about gradients later, and this idea of derivatives - don’t worry. I’m not saying it’s not hard, I’m saying we’ll dive deeper into it later.

We won't write raw PyTorch in this lesson. But everything we do today runs on PyTorch underneath, and starting in L3 you'll be writing it directly - building training loops, computing losses, and running gradient descent by hand. Understanding that PyTorch is the engine under the hood is important context.

For a quick 2-minute overview: [Fireship - PyTorch in 100 Seconds](https://www.youtube.com/watch?v=ORMx45xqWkA)

For a deeper dive into tensors and autograd: [PyTorch official autograd tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

## Why fastai? (and why we'll stop using it)

fastai is a high-level library built on top of PyTorch. It handles a lot of the boilerplate - data loading, augmentation, training loops, learning rate scheduling - so you can get a model working in just a few lines of code.

Think of it like this: PyTorch is like building a car from parts. fastai is like driving one. You need to learn to drive before you learn to build engines. That's why we start here.

**The benefits of starting with fastai:**

- You see the full ML process in one lesson without getting lost in implementation details
- You get a working model fast - that motivation matters when you're learning
- It introduces the right concepts (loss, epochs, overfitting, augmentation) in context - you see them in the training output before you have to implement them
- The code is readable enough to follow what's happening at a high level

**Why we won't keep using it:**

- fastai hides too much. When `fine_tune(3)` runs, it's doing about 50 things behind the scenes - freezing layers, adjusting learning rates, running the training loop, computing gradients. If you don't know what those things are, you can't debug problems or adapt to new situations
- In the real world, most teams and research labs use pure PyTorch. Reading papers, understanding other people's code, contributing to open source - all requires PyTorch fluency
- When something goes wrong (and it will), you need to understand the internals to fix it. fastai is great until it isn't, and then you're stuck
- You can't truly understand ML concepts like backpropagation, loss landscapes, or regularization if a library is doing them invisibly

This is the only lesson that uses fastai. Starting in Lesson 3, we switch to pure PyTorch and build everything ourselves - the training loop, the loss calculation, the gradient updates. You'll understand every line of code. fastai is the quick win that shows you the destination before we learn the route. But fastai will be great to get started, we’ll get a super good feel for the process of training a model. Having that said, don’t focus too much on memorizing fastai syntax, just try to focus on the process and the steps we seem to be doing (exploring, preprocessing, training, evaluation, etc).

For a deeper understanding of fastai's DataBlock API (the data loading system we'll use): [fastai DataBlock tutorial](https://docs.fast.ai/tutorial.datablock.html)

For the full fastai vision tutorial: [fastai Computer Vision intro](https://docs.fast.ai/tutorial.vision.html)

## Transfer Learning: A Shortcut We'll Use

One practical detail worth mentioning: we won't train a model from scratch. Training an image classifier from zero requires millions of images and days of compute. Instead, we use a pretrained model (ResNet34) that someone already trained on 1.2 million images. It already knows how to detect edges, textures, and shapes. We just teach it our 37 pet breeds on top of that existing knowledge.

This is called **transfer learning**, and `learn.fine_tune(3)` is the fastai shortcut that does it. We'll come back to transfer learning properly in later lessons - for now just know it's why we can get 90% accuracy with only ~7,000 images instead of millions. In upcoming lessons, we’ll train image models from scratch, but just for today we’ll fine-tune a model - the process will be very similar to a standard process, so don’t worry too much about it.

## Key Concepts Introduced

These concepts show up naturally during the notebook. We don't front-load definitions - they get explained when you see them in the training output.

**Epoch** - one complete pass through all training data. If you have 5,000 training images and train for 3 epochs, the model sees each image 3 times. More epochs = more chances to learn, but also more risk of overfitting.

**Loss** - a single number measuring how wrong the model's predictions are. Lower is better. In L1 we saw loss visually (the gap between the line and the data points in the regression demo). Here you'll watch the loss number drop during training. The specific loss function we use is called cross-entropy loss - we'll derive it from scratch in L3, for now just know: it measures wrongness.

**Accuracy** - the percentage of validation images the model classifies correctly. This is the metric we actually care about. Loss is what the model optimizes internally, accuracy is what we report.

**Training set vs validation set** - we split the data into two parts. The model trains on the training set (learns patterns). We evaluate on the validation set (checks if it actually generalized). If accuracy is high on training data but low on validation data, the model memorized instead of learned - that's overfitting.

**Learning rate** - controls how aggressively the model adjusts its weights during training. Too low: the model learns painfully slow. Too high: it overshoots and the loss goes haywire. Just right: smooth convergence. Finding a good learning rate is one of the most important practical skills in ML. fastai has a built-in learning rate finder that helps.

**Overfitting vs underfitting** - the central tension in ML:

- Underfitting: model is too simple or hasn't trained enough. Both training and validation loss are high.
- Good fit: model captures real patterns. Both losses are low and close together.
- Overfitting: model memorized the training data. Training loss is very low but validation loss starts rising.

Google has a nice interactive explanation of overfitting: [Google ML Crash Course - Overfitting](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting)

**Data augmentation** - randomly transforming training images (flipping, rotating, changing brightness) so the model sees slightly different versions each epoch. This prevents it from memorizing specific images and forces it to learn the actual patterns. Think of it as making the training data artificially larger and more diverse.

**Confusion matrix** - a grid showing which classes the model confuses with each other. If the model often predicts "Ragdoll" when the answer is "Birman", those two breeds appear as a hot spot in the confusion matrix. Very useful for understanding where the model struggles. For a deeper dive: [Machine Learning Mastery - What is a Confusion Matrix](https://machinelearningmastery.com/confusion-matrix-machine-learning/)

## What We'll Do in the Notebook

### Step 1: Understand Your Data

Load the Oxford-IIIT Pet Dataset (37 breeds, ~7,400 images). Look at the data structure, check the class distribution, visually inspect sample images. This step is crucial and often skipped - you need to know what you're working with before training anything.

### Step 2: Prepare the Data

Set up the data pipeline using fastai's DataBlock API. This handles:

- Splitting into training (80%) and validation (20%) sets
- Resizing images to a consistent size (224x224)
- Applying data augmentation (random flips, rotations) to training images only
- Batching images into groups of 64 for efficient GPU processing

### Step 3: Train the Model

Load a pretrained ResNet34 and fine-tune it on our pet data. Watch the training output - you'll see epoch number, training loss, validation loss, and accuracy updating in real time. We also visualize loss curves and discuss what they tell us about model health.

### Step 4: Evaluate

Look at the confusion matrix to see which breeds get mixed up. Examine the model's biggest mistakes (top losses). Make predictions on individual images and see the confidence scores. Peek under the hood at what the model actually receives as input (a 224x224x3 grid of numbers).

### Step 5: Iterate

Experiment with different settings: more epochs, a bigger architecture (ResNet50), larger images (320px instead of 224px). Change one thing at a time, measure the impact. This is how real ML development works - systematic experimentation.

### Step 6: Ship

Export the trained model to a .pkl file. Load it back and prove it works without the training code. Sketch a deployment architecture (FastAPI backend + Streamlit frontend) to show that ML models don't have to live in notebooks.

## Terminology

| Term | What it means | Where we see it |
| --- | --- | --- |
| **Epoch** | One complete pass through all training data | Training output: "epoch 1 of 3" |
| **Loss** | How wrong the model is (lower = better) | Training output: loss column dropping |
| **Accuracy** | % correct predictions on validation set | Training output: accuracy column rising |
| **Training set** | Data the model learns from | 80% of images |
| **Validation set** | Held-back data for checking generalization | 20% of images |
| **Transfer learning** | Starting with a pretrained model | ResNet34 trained on ImageNet |
| **Fine-tuning** | Adapting pretrained model to new task | `learn.fine_tune(3)` |
| **Data augmentation** | Random transforms to prevent memorization | Flips, rotations, brightness |
| **Overfitting** | Model memorized training data | Validation loss starts rising |
| **Learning rate** | Step size for weight adjustments | Controls training speed |
| **Confusion matrix** | Grid of predicted vs actual classes | Where the model gets confused |
| **Batch** | Group of images processed together | 64 images at a time |

## Connection to L1 and L3

**From L1:** We introduced features, labels, training, inference, loss, gradient descent, and the idea that models are just collections of numbers (weights) that get optimized. L2 builds on all of this - you'll see loss in action (watching it drop), you'll see the train/test split applied, and you'll see a model with 21 million weights doing something impressive.

**To L3:** Everything fastai does in one line, we'll build by hand in L3. The loss function? We'll derive it. The training loop? We'll code it. Gradient descent? We'll implement it step by step. L2 shows you the destination. L3 starts building the road.

## Resources

### Before the lesson

- [fast.ai](http://fast.ai/) Practical Deep Learning - Lesson 1 (~90 min, same top-down philosophy we use, covers transfer learning and the fastai workflow in depth): https://course.fast.ai/Lessons/lesson1.html

### Documentation and tutorials

- fastai Computer Vision tutorial (official docs, walks through image classification step by step): https://docs.fast.ai/tutorial.vision.html
- fastai DataBlock tutorial (official docs, explains the data loading system we use in the notebook): https://docs.fast.ai/tutorial.datablock.html
- fastai Vision Learner docs (reference for the learner object and fine_tune): https://docs.fast.ai/vision.learner.htm
- Google ML Crash Course - Overfitting (interactive explanation with visual examples): https://developers.google.com/machine-learning/crash-course/overfitting/overfitting

### Quick videos

- IBM Technology - What is Transfer Learning? (~5 min, clear explanation with examples): https://www.youtube.com/watch?v=BqqfQnyjmgg
- deeplizard - Fine-tuning a Neural Network Explained (short, focused on the concept of fine-tuning): https://deeplizard.com/learn/video/5T-iXNNiwIs

### What is a neural network? → We’ll dive deeper into this upcoming lessons (l4 and onward), but feel free to start checking it out. It might feel super hard, and that’s fine, again - we’ll dive deeper later, and I’ve spent a lot of time on the notebooks 4,5,6,7,8~ to help explain it.

- 3Blue1Brown - But What is a Neural Network? (19 min, excellent visual intuition for how neural networks work, great complement to our neural network intro from L1): https://www.youtube.com/watch?v=aircAruvnKk
- StatQuest - Neural Networks Pt.1 (20 min, builds up from simple models to neural nets, clear and methodical): https://www.youtube.com/watch?v=CqOfi41LfDw

### The fastai course

The fastai course by Jeremy Howard is a solid companion to this course. It has a similar top-down philosophy (use it first, understand later). Their first two lessons overlap with our L1-L2. The course is free and available at https://course.fast.ai/. Worth watching the first lesson before or after our L2. In fact, this course is very much inspired by the fastai course, but the goal is to do everything in pure pytorch instead of using the fastai framework, and we’ll spend more or less time on certain topics.

- [fast.ai](http://fast.ai/) Practical Deep Learning - Lesson 2 (deployment and production, goes deeper on the ship step): https://course.fast.ai/Lessons/lesson2.html
- Dive into Deep Learning - Fine-tuning chapter (textbook-style explanation with code): http://d2l.ai/chapter_computer-vision/fine-tuning.html
- Codecademy - Getting Started with PyTorch (article, beginner-friendly intro): https://www.codecademy.com/article/getting-started-with-pytorch-a-beginners-guide-to-deep-learning