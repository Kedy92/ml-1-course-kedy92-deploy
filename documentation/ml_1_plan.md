- [ ]  L7 Fastai
- [ ]  CNN’s
- [ ]  L8 - L12
- [ ]  L13 deployment?
- [x]  Collaborative filtering + Embeddings (moved to Bonus Lessons)
- [ ]  LLM:s

[My ML experience](https://www.notion.so/My-ML-experience-30f6064241a180a0a9e1c18dcb11161e?pvs=21)

[Ideas](https://www.notion.so/Ideas-2b56064241a180b889acf71de8996375?pvs=21)

[Planering (1)](https://www.notion.so/2ec6064241a18038ad94cc4ecd2e67ca?pvs=21)

## Resources

Decent tutorial on pytorch with some good videos:

https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4

# ML Course 1 - Detailed Plan

**Audience:** Students with Python/web dev background (know FastAPI, Docker, PostgreSQL)

**Framework:** PyTorch is the primary deep learning framework. fastai appears in L2 as a high-level tool to showcase end-to-end ML before students understand the internals. L3-L4 build models from scratch in NumPy first (so students see the raw math), then transition to PyTorch. From L5 onwards, PyTorch is the default for neural networks. sklearn is used throughout for preprocessing, evaluation, and tree-based models (L7).

---

## Quick Overview

Possible changes:

- Remove one CNN-lesson

| Lesson | Focus |  |
| --- | --- | --- |
| L1 | The Hook: Tour of ML |  |
| L2 | The Hook: First Model (fastai) |  |
| L3 | Foundations: Regression + Gradient Descent |  |
| L4 | Foundations: Classification → Neural Nets (Top-Down) |  |
| L5 | MLPs: Forward Pass + Training Loop |  |
| L6 | MLPs: Full Tabular Pipeline |  |
| L7 | Trees: Random Forests & Gradient Boosting |  |
| L8 | Project Assistance (no new content) |  |
| L9 | Images: Complete Pipeline with MLP |  |
| L10 | Images: MLP Practice (CIFAR-10) |  |
| L11 | CNNs: Concept & Architecture |  |
| L12 | CNNs: Fine-Tuning & Deployment |  |
| L13 | Transfer Learning & Projects |  |
| L14-L19 | Modern Stack (LLMs, Embeddings, RAG) |  |

---

## The Hook (Top-Down)

### Lesson 1: Lay of the Land

**Focus:** Show what ML can do, introduce core terminology.

**Topics:**

- Tour: working demos across data types (tabular, images, text)
- What is supervised learning? Input → Model → Output
- Core terminology: model, training, inference, features, labels
- Brief mention: trees, neural networks, transformers exist
- Where we're going: "By week 6, you'll build and deploy your own classifier"

**Demos:**

- sklearn on Titanic (tabular classification) + California Housing (regression)
- torchvision ResNet on images + YOLO object detection
- HuggingFace pipeline on text (sentiment analysis, text generation)

**Terminology introduced:** features, labels, model, training, inference, supervised learning, classification, regression, overfitting, train/test split, parameters/weights, gradient descent, epoch, loss

**Note — "The overwhelm talk":** Include a moment (ideally early) where we acknowledge that ML has a brutal learning curve. There are so many moving parts — terminology, math, code patterns, conceptual layers — that it can feel like your brain is short-circuiting. Generate/include an image of a chaotic, overloaded brain (symbolizing frustration, information overload). Use it to normalize the feeling and deliver the message: give things time to sink in. Repetition is how it clicks — concepts that feel impossible in week 2 become obvious by week 6. You don't need to understand everything to be productive. "Släpp sargen" — let go of the need to master every detail before moving forward. The course is designed so that you'll revisit every core idea multiple times, from different angles, and each pass adds clarity. Consider adding this to the L1 notebook as well.

### Lesson 2: Your First Model (Quick Win)

**Focus:** Train and fine-tune a vision model using fastai. See it work before understanding why.

**Topics:**

- Structured around a 6-step ML process: Understand -> Prepare -> Train -> Evaluate -> Iterate -> Ship
- fastai DataBlock API: loading image data
- Transfer learning: use a pretrained ResNet
- Fine-tuning: `learn.fine_tune()`
- See the output: loss going down, accuracy going up
- Interpret results: confusion matrix, top losses analysis
- Iterate: disciplined experimentation (bigger model, more epochs, higher resolution)
- Deploy: model export, Streamlit + FastAPI app, Docker Compose
- "It works! But how? That's what we'll learn next."

**Terminology introduced (as they appear in output):**

- epoch ("we're on epoch 2 of 3")
- loss ("see this number going down?")
- learning rate ("this controls step size")
- train/valid ("why two losses?")
- accuracy ("how often it's right")

**Dataset:** Pet breeds (37 breeds, Oxford-IIIT Pets)

**The point:** Students leave with a working image classifier. Curiosity is sparked. Week 2 explains how it actually works.

---

## Foundations (How Models Learn)

Resources:

The google made tutorials can be a good crash course into linear regression:

https://developers.google.com/machine-learning/crash-course/linear-regression

### Lesson 3: Regression

**Focus:** Introduce gradient descent - the engine that powers all learning.

**Topics:**

- Problem: predict continuous value (house prices)
- Linear regression: y = wx + b
- Loss function: MSE - "how wrong are we?"
- Gradient descent: "follow the slope downhill"
    - Visual: loss landscape, moving toward minimum
    - Math: compute gradient, update weights
    - Learning rate: step size matters (interactive exploration)
- Multiple features: "same thing, just dot product"
- The training loop: forward -> loss -> backward -> update -> repeat
- Code from scratch in NumPy, then show PyTorch version (validate they match)
- Evaluation metrics: MSE, RMSE, MAE, R²
- Overfitting vs underfitting: polynomial regression showing the progression
- Multiple features: adding features, feature importance, diminishing returns
- Neural network teaser: show a simple NN beating linear regression on same data, preview L4

**Terminology introduced:** parameters/weights, loss function, gradient, gradient descent, forward pass, learning rate, hyperparameters, epoch, overfitting, underfitting, extrapolation

**Dataset:** California Housing

---

### Lesson 4: Classification + Neural Networks (Top-Down)

Resources:

Googles tutorial on logistic regression is a decent place to start. A lot of the basic classification problems have started with logistic regression. However, it’s not very detailed, and no proper video content.

https://developers.google.com/machine-learning/crash-course/logistic-regression

**Focus:** Build the argument for neural networks. Show they work before explaining how.

**Topics:**

- Opening demos: classification across domains (medical diagnosis, fraud detection, handwritten digits) using sklearn black boxes to show the breadth of classification
- Problem shift: predicting categories, not numbers
- Titanic dataset: data exploration, survival patterns by gender/class/age
- Manual decision boundary: try drawing a line by hand, see it fail (~67%)
- Linear regression on binary data: shows why it breaks (outputs outside 0-1)
- Sigmoid: "squash output to probability 0-1"
- Logistic regression: linear + sigmoid
    - Same gradient descent loop from L3, built from scratch in NumPy
    - Cross-entropy loss: full "Why Not MSE?" section with interactive comparison
    - Train on Titanic, see accuracy
    - Weight interpretation: model discovers "women and children first" from data alone
- Decision boundary visualization: straight line on 2D scatter plot
- Overfitting in classification: polynomial features (degree 1/3/8) showing underfit/fit/overfit
- The limitation: manual feature engineering doesn't scale
- Enter the MLP (PyTorch):
    - PyTorch intro: tensors, automatic differentiation, nn.Module
    - sklearn MLPClassifier shown for reference (quick aside)
    - MLP architecture: one hidden layer with 16 neurons
    - Explicit connection: sigmoid and cross-entropy reused from logistic regression, only new piece is hidden layer + ReLU
    - Train on same Titanic data
    - Compare decision boundaries: straight (logistic) vs curved (MLP) side by side
- Conceptual explanation:
    - Hidden layers: "intermediate representations" / "automatic feature engineering"
    - ReLU: "non-linearity between layers" (light intro, L5 goes deeper)
- Preview: "It works better. But what's actually happening? Next lesson."

**Terminology introduced:** sigmoid, cross-entropy, logistic regression, decision boundary, activation function, hidden layer, ReLU, neurons, layers, architecture, tensor, automatic differentiation

**Dataset:** Titanic

---

## MLPs (Multilayer Perceptrons)

Resources:

https://medium.com/@sahin.samia/train-a-neural-network-in-pytorch-a-complete-beginners-walkthrough-3897d18d6078

### Lesson 5: First MLP - The Training Loop

**Focus:** Understand what happens inside a neural network, especially the forward pass.

**Topics:**

- "Remember the MLP from L4? Let's understand what each line does"
- Forward pass (main focus):
    - Input data as tensor
    - First linear layer: matrix multiplication + bias
    - ReLU activation: what it does to the values
    - Second linear layer: another matmul + bias
    - Sigmoid: final probability output
    - "Trace your data through the network"
- Loss calculation: comparing predictions to targets
- Backpropagation (simplified):
    - "We calculate partial derivatives"
    - "This tells us how much each weight affected the loss"
    - "PyTorch does this with `loss.backward()`"
    - Gradients are stored, ready to use
- Weight update: `weight -= lr * gradient`
- The full loop: forward → loss → backward → update → repeat
- Epochs and batches: "process data in chunks, repeat multiple times"
- PyTorch basics woven in: tensors, requires_grad, nn.Module
- Optimizers: "SGD is what we've been doing manually, Adam is the common default"

**Terminology introduced:** tensor, batch, epoch, backpropagation (simplified), optimizer, SGD, Adam

**Dataset:** Titanic (continue from L4, now understanding what's inside)

---

### Lesson 6: Full Tabular Pipeline

https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html A tutorial using an image model, but you can still follow the pipeline somewhat. 

Dataset & dataloaders 

https://www.youtube.com/watch?v=PXOzkkB5eH0

Others:

https://www.slingacademy.com/article/pytorch-classification-from-scratch-building-a-dense-neural-network/

Overall tutorials:

https://yashuseth.wordpress.com/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/

Or this one

https://francescopochetti.com/pytorch-for-tabular-data-predicting-nyc-taxi-fares/

Detailed but uses lightning pytorch 

https://medium.com/@david.tiefenthaler/pytoch-lightning-tabular-classification-312d6b753d28

**Focus:** Complete end-to-end workflow for training on tabular data. (Similar to Jeremy's L5)

**Topics:**

- Data exploration: understand what you're working with
- Data preparation:
    - Missing values: checking, strategies for handling
    - Normalization: why neural nets need it, standard scaling
    - Categorical encoding: one-hot vs label encoding, when to use which
    - Log transform for skewed distributions
    - Train/validation split
- Build the full pipeline:
    - Preprocessing → DataLoader → Model → Training loop → Evaluation
- Evaluation:
    - Metrics: accuracy, precision, recall, confusion matrix
    - Cross-validation: "more robust than single split" (brief intro)
    - Reading the results: what do they tell us?
- Class imbalance: "what if 95% of data is one class?" (brief intro)
- "We've trained an MLP on tabular data. But is this the best tool for tabular?"

**Terminology introduced:** train/test split, confusion matrix, precision, recall, cross-validation, class imbalance

**Dataset:** New tabular dataset

---

### Lesson 7: Trees - Random Forests & Gradient Boosting

Notebooks: 

- modern_tabular_ml.ipynb
- how random forests really work

Resources:

Random forests:

https://www.youtube.com/watch?v=AdhG64NF76E&list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU&index=7

Gradient boosted trees (make sure to first work with decision trees and random forests before this - it’s not essential to understand GBT’s, it’s a lot trickier to understand the process as its sequential:

https://www.youtube.com/watch?v=PxgVFp5a0E4

Model interpretation using SHAP:

https://www.youtube.com/watch?v=MQ6fFDwjuco

Essentially,

1. Showcase classification using titanic
2. Showcase regression with bulldozer example
3. Don’t go into detail on how gradient boosted trees work
4. More concise notebook with practical project
5. Possibly do something more practical with the model

**Focus:** Alternative to neural nets for tabular data. Honesty: "trees often win here."

**Topics:**

- Decision trees:
    - How they work: binary splits, impurity (Gini)
    - Visual: tree structure, decision boundaries
    - Pros: interpretable, no preprocessing needed
    - Cons: overfits easily
- Random Forests:
    - Ensemble of trees, bagging
    - Why averaging works (uncorrelated errors cancel)
    - Feature importance
- Gradient Boosting (brief):
    - XGBoost, LightGBM, CatBoost
    - "Trees that learn from mistakes"
    - Often state-of-the-art on tabular
- Comparison on same dataset:
    - MLP vs Random Forest vs Gradient Boosting
    - Discuss results honestly
- When to use what:
    - Tabular: trees often win, try both
    - Images/text: neural nets
- MLP repetition: rebuild and compare, reinforce understanding

**Terminology introduced:** decision tree, random forest, bagging, ensemble, gradient boosting, feature importance

**Dataset:** Titanic (compare to L6 MLP results) + possibly a second dataset

---

### Lesson 8: Project Assistance

**Focus:** No new content. Students have been working on assignments from weeks 1-4. This lesson is dedicated to helping students with their work, answering questions, and reviewing progress before diving into image recognition.

**Why a break here:** Students need time to consolidate tabular ML (MLPs + trees) before switching to a fundamentally different data domain. Rushing into images without a solid tabular foundation creates confusion.

## Images & CNNs

### Lesson 9: Images — The Complete Pipeline with MLP

**Focus:** Teach the full image classification pipeline end-to-end. Same MLP architecture from tabular, applied to images. Demonstrate everything CIFAR-10 hides from you.

**Topics:**

- Images as data: pixels, RGB channels, flattening
- Data cleaning: verify images load, handle corrupted files
- Resizing strategies: squash vs crop vs resize+crop
- `transforms.Compose()` pipeline (compare to pandas preprocessing)
- Per-channel normalization: compute mean/std per RGB channel (like StandardScaler)
- Data augmentation: flip, rotation, color jitter, random crop
- Visual verification: always check your pipeline before training
- Multi-class classification: softmax + CrossEntropyLoss (traced with real numbers)
- MLP architecture: flatten + linear layers, parameter explosion
- LR finder, training loop with model.train()/eval()
- Evaluation: confusion matrix, per-class accuracy, top losses analysis
- MLP limitations: pixel shuffle experiment (zero spatial awareness)
- CNN teaser: fewer params, better accuracy, spatial awareness

**Terminology introduced:** softmax, multi-class classification, data augmentation, per-channel normalization, transforms pipeline, top losses

**Dataset:** Bird photos (5 classes: eagle, flamingo, owl, parrot, penguin) — web-scraped, messy, forces every pipeline step

**Notebook:** `08_v2_mlp_image_classification.ipynb`

### Lesson 10: Images — MLP Practice (CIFAR-10)

**Focus:** Students practice image classification independently on CIFAR-10. Experiment with architectures, hyperparameters, regularization.

**Topics:**

- Apply pipeline from L9 to a new dataset (CIFAR-10, 10 classes)
- Architecture experimentation: width, depth, activation functions
- Hyperparameter tuning: learning rate, batch size, epochs
- Regularization: dropout, weight decay, batch normalization
- Reading loss curves: diagnosing overfitting vs underfitting
- Compare multiple model variants
- Observe MLP ceiling on images: "We've pushed it as far as it goes"

**Terminology introduced:** learning rate scheduler, weight decay, regularization strategies

**Dataset:** CIFAR-10 (10 classes, 32x32 RGB)

**Notebook:** `08_mlp_image_classification_project.ipynb`

### Lesson 11: CNNs — Concept & Architecture

Resources:
A decent intro:
https://www.youtube.com/watch?v=pj9-rr1wDhM 
Or this one:
https://www.youtube.com/watch?v=HGwBXDKFk9I
A more practical example using excel, although the example is a bit small unfortunately and doesnt really showcase it using pytorch - From minute 44~, where he talks about convolutions:
https://www.youtube.com/watch?v=htiNBPxcXgo


**Focus:** Why CNNs work for images, understand convolutions, build and train a CNN from scratch in pure PyTorch.

**Topics:**

- Why MLPs struggle: parameter explosion + no spatial awareness (recap from L9)
- Convolution operation: kernels/filters sliding over image
- Feature maps: what kernels detect, how to read them
- Channels: how kernels grow a depth dimension to match input channels (RGB, then learned features)
- Stride 2: shrinking spatial dimensions, replacing max pooling
- The architecture pattern: spatial shrinks, channels grow, funnel to prediction
- Build CNN from scratch in PyTorch (`nn.Conv2d`, custom `conv` helper)
- Training stability: diagnosing with activation statistics, batch normalization, 1cycle LR scheduling
- Compare: CNN vs MLP on same dataset (fewer params, higher accuracy)
- Brief overview of famous architectures: ResNet (skip connections), EfficientNet (conceptual only)

**Dataset:** CIFAR-10 (32x32 color, 10 classes) or birds from L9 — compare directly to MLP results

**Notebook:** Pure PyTorch, no fastai. Teaching notebook style.

### Lesson 12: Fine-Tuning, Object Detection & Deployment

**Focus:** The practical "real world" image lesson. Fine-tune pretrained models, introduce object detection with YOLO, deploy via FastAPI. Closes the image section.

**Topics:**

Part 1 — Fine-tuning pretrained models:
- Why train from scratch when someone already did?
- Pretrained models in torchvision (ResNet, EfficientNet)
- Feature extraction: freeze backbone, train new head
- Fine-tuning: unfreeze layers, lower LR
- Fine-tune on custom dataset, compare accuracy vs from-scratch CNN (dramatic difference)

Part 2 — Object detection with YOLO:
- Classification vs detection: "what is this?" vs "where is everything?"
- YOLO as a practical tool (focus on usage, not internals)
- Load pretrained YOLOv8, run inference on images
- Fine-tune YOLO on a custom dataset (e.g. defect detection, document elements)
- Why this matters: most common commercial CV task (quality control, warehouse, retail, documents)

Part 3 — Deployment:
- Save model: `torch.save()`, `state_dict()`
- FastAPI endpoint: accept image → preprocess → predict → return
- Containerize with Docker
- Deploy to AWS EC2 instance
- Production considerations: input validation, error handling

**Outcome:** Students fine-tune a pretrained classifier AND a YOLO detector, then deploy one as a working API

**Reference:** `/home/ua-tobias/projects/ml_projects/modified/model_api` (prior deployment project)

---

## Modern Stack (LLMs, Embeddings, RAG)

*Detailed planning TBD - rough outline below*

### Current Outline

| Lesson | Topic | Notes |
| --- | --- | --- |
| L13 | LLM APIs + structured outputs | OpenAI/Claude APIs, JSON mode, Pydantic validation |
| L14 | Embeddings + semantic similarity | HuggingFace sentence-transformers vs API embeddings, visualizing with PCA |
| L15 | Vector databases (pgvector) | Students know PostgreSQL already |
| L16 | RAG fundamentals | Query → embed → retrieve → generate |
| L17 | RAG improvements | Chunking strategies, reranking concepts |
| L18 | Final project + deployment | Deploy RAG system with FastAPI |

### L14: Embeddings + Semantic Similarity (Expanded)

**Topics:**

- What are embeddings? Dense vector representations of text
- API embeddings vs local models (HuggingFace sentence-transformers)
- Generating embeddings for sentences/documents
- Semantic similarity: cosine distance between vectors
- "Similar meaning = close in vector space"
- Visualizing embeddings with PCA:
    - "768 dimensions - how do we see this?"
    - PCA reduces to 2D/3D for plotting
    - Scatter plot: see similar items cluster together
    - "This is why semantic search works"
- Practical: embed a set of documents, visualize, find similar pairs

**Terminology introduced:** embeddings, semantic similarity, cosine similarity, PCA (as visualization tool)

### Brief Historical Context for L14

Before diving into LLMs, briefly mention:

- **RNNs (Recurrent Neural Networks):** "Before transformers, RNNs processed sequences one step at a time. Good for text, but slow and forgot long-range context."
- **Attention mechanism:** "Transformers use attention to look at all parts of input at once. This is why LLMs work so well."
- Keep it to 5-10 minutes, not a deep dive. Just enough context to appreciate why transformers/LLMs are significant.

### Other Lesson Ideas (Brainstorming)

- LLM intro lesson: How transformers work (high-level)
- Tokenization deep-dive: BPE, vocabulary, token limits
- Agent lesson: Building agents with tool use
    - Could lead to final project: grading system using agents
    - Include prompt injection awareness
- Guest lecture: Using LLM APIs in production
- Fine-tuning LLMs (might be Course 2)

**Prompt engineering:** Sprinkled throughout course via agentic coding practice (Claude Code, Cursor). Key skills: context files ([CLAUDE.md](http://claude.md/)), spec-first approach, plan-then-implement pattern. No dedicated lesson needed.

**HuggingFace integration:** L15 introduces HuggingFace ecosystem through sentence-transformers. Students see the tradeoff: API embeddings (simple, cost) vs local HF models (free, more control). Fine-tuning transformers is Course 2.

---

## Bonus Lessons

Standalone topics that enrich the course but aren't part of the main progression. Can be taught between blocks, as Friday sessions, or as self-study material.

### Collaborative Filtering & Recommendation Engines

**Primary resource:** [Jeremy Howard — Practical Deep Learning, Lesson 7](https://www.youtube.com/watch?v=p4ZZq0736Po&list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU&index=7) (start from 1:02:00). Arguably the best lesson in the entire series — Jeremy explains collab filtering clearly using PyTorch, stepping through the math in a spreadsheet before coding. The way he builds from Excel SGD to embeddings to a full neural net version is exceptionally well done.

**Why it matters:** Bridges tabular ML (L6) and the modern stack (L14+). Students see embeddings as learned representations before encountering them in NLP/LLM context. Also demonstrates that SGD + loss works outside neural networks.

**Topics:**
- The recommendation problem: sparse user-item matrix, predict missing ratings
- Latent factors: randomly initialized embedding vectors for users AND items — both are learnable parameters, not inputs
- Dot product as similarity: no hidden layers, no ReLU, purely linear — and it still works because it's matrix factorization, not function approximation
- Biases: some users rate high, some movies get rated high regardless of match
- Weight decay / regularization to prevent overfitting
- Interpreting embeddings: PCA to visualize learned movie clusters
- `CollabNN`: upgrading to a real neural network with hidden layers for nonlinear interactions
- Cold start problem: new users/items have random embeddings, need content features to bootstrap

**Key concept — latent factors vs content features:**
- **Latent factors only**: just embeddings + dot product. Discovers hidden patterns from ratings alone. Can't handle new users/items with no history.
- **Content features only**: traditional ML on known attributes (genre, age, etc.). Handles cold start but misses subtle taste patterns.
- **Hybrid** (what Netflix/Spotify/YouTube do): embeddings for IDs + real features + neural network on top. Best of both worlds.

**Notebook:** `lessons/recommendation_engines/collab_filtering_fastai.ipynb`

**Notes:** `lessons/recommendation_engines/l7_collab_filtering_2026_02_23.md`

## Optional: Project Week

No lessons. Students work independently on final project.

---

## Potential Topics to Add

Topics that may need dedicated lessons or integration into existing lessons. Based on comparison with FastAI, Andrew Ng's course, CMU 10-601, and common industry syllabi.

### High Priority

**1. Embeddings & Collaborative Filtering** ✓ Moved to Bonus Lessons section

**2. Text Basics / Tokenization**

- Currently: Jump straight to LLM APIs in L14
- Problem: Students don't understand how text becomes numbers
- Recommendation: Brief intro (15-20 min) at start of L14
- Teaches: Tokenization, vocabulary, token limits, why text needs special handling

### Integrated Into Curriculum ✓

These concepts are introduced where they naturally fit, then used throughout subsequent lessons.

| Topic | First Introduced | Then Used In |
| --- | --- | --- |
| Optimizers (SGD, Adam) | L5 (training loop) | Every lesson after |
| Softmax (multi-class) | L9 (10-class images) | L11, L12, anywhere multi-class |
| Data augmentation | L9 (intro), L10 (expand) | Image lessons |
| Learning rate schedulers | L10 | L11+ when training |
| Batch normalization | L11 | CNN architectures |
| Cross-validation | L6 | When evaluating models |
| Class imbalance | L6 | When relevant to dataset |
| Dropout, weight decay | L10 | When regularizing |
| RNNs (brief context) | L14 | Historical mention before LLMs |

### Consider Adding

**PCA (Principal Component Analysis)** ✓ Added to L15

- Used for visualizing embeddings (768-dim → 2D scatter plot)
- Practical tool, not deep theory - students see it in action

### Skip (Like FastAI Does)

**Classical ML (SVM, KNN, Naive Bayes)**

- Trees cover the "alternative to neural nets" niche
- These algorithms add less practical value for this course's goals

**Attention Mechanism Deep Dive**

- Can use LLMs effectively without understanding internals
- Course 2 material if we cover fine-tuning transformers

---

## Teaching Philosophy

**Principles:**

- Top-down: show it working, then explain
- Theory + code together: never pure lecture
- Hands-on: students code every lesson
- Practical: deployment matters, not just notebooks
- Honest: "trees beat neural nets on tabular" - right tool for the job

**Narrative Arc:**

```
L1-L2:   "Here's ML, look what's possible" (hook)
L3-L4:   "Here's how models learn" (gradient descent, neural nets intro)
L5-L7:   "Here's tabular ML in depth" (MLPs + trees, when to use what)
L8:      Project catch-up (consolidate tabular before images)
L9-L10:  "Here's how to handle images with MLPs" (pipeline + practice)
L11-L12: "CNNs + fine-tuning + deployment" (ship an image classifier)
L13+:    "Here's the modern LLM stack"

```

---

*Created: 2025_01_18_1630Updated: 2025_01_25_1900*

## Random notes

- I should use some kind of auto grading tool like G did.
- I need to consider what the assignments should look like from week 1 to week 6. Probably autograder as final assignment, or optional project.

## Resources

Model interpreteble vs explainable

https://www.youtube.com/watch?v=VY7SCl_DFho