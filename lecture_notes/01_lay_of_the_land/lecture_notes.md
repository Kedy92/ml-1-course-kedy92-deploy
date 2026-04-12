This lesson is a flyover of the machine learning landscape. The goal isn't to understand how anything works yet, it's to see what's possible, learn the vocabulary, and build a mental map of the field that every future lesson will fill in. This course will have a top-down approach - it means we’ll start off working very practical, and then we’ll start going into depth of the concepts in future lessons more (starting lesson 3). Lesson 1-2 are heavily focused on just trying things out and seeing the magic of machine learning.

### A quick word on learning machine learning

If you’ve got a coding background learning machine learning will still be hard, but it really helps that you know coding. But learning ML can really feel like you are introduced with a wall of new concepts and terminology, and it’s going to take a while to get used to all the new stuff. You’ll mix things up, and that’s completely fine. Learning ML is a long process, probably a year at least to get somewhat comfortable and not constantly second guessing things.

Now, that’s one thing. Another painful part is that it’s always difficult to know how deep you should try to understand something. My advice here is that you don’t have to dive too deep into details, instead, remember that your goal is to be productive but still have a good feel for the process. And so, you might feel like you really want to understand the chain rule and backpropagation, but sometimes these really difficult concept requires us to let go. Yepp, LET GO! Don’t obsess over understanding how everything exactly works - maybe it’s sometimes enough to just understand “if I use this weird thing called batch normalization, that will help my machine learning model and reduce overfitting”. Now, you could deepdive into what “batch normalization” is, but you could also just accept that it’s a cool thing some smart person invented in a scienctific white paper and everyone just seems to use it.

![bra.png](attachment:eed676a6-23ac-4d27-a83d-c51af0487ce7:bra.png)

In swedish, we say “släpp sargen”, or “kom in i matchen”, what we mean is: let go, accept that some stuff is magical and it just works. And realize that your goal is ultimately to be productive with using ML - deeper knowledge will come in time. But you should absolutely try to understand a lot of the concepts, you might just not need to understand the math or details behind them at all times.

![Gemini_Generated_Image_zeb3l0zeb3l0zeb3.png](attachment:93d24a0b-96c7-46b3-9ac4-0c53fef4f1d7:Gemini_Generated_Image_zeb3l0zeb3l0zeb3.png)

## What is Machine Learning?

In traditional programming, you write rules: "if the email contains these words, it's spam." Machine learning flips that - instead of writing rules, you show the system thousands of examples of spam and not-spam, and it figures out the rules on its own. The rules it discovers are often subtler and more effective than anything a human would write by hand.

| Traditional Programming | Machine Learning |
| --- | --- |
| You write the rules | The algorithm discovers the rules |
| Rules + Data = Answers | Data + Answers = Rules |
| Breaks when patterns change | Retrain on new examples |
| Hard to scale to complex problems | Scales with data |

Here's what that looks like in practice. Say you want to predict whether an email is spam:

| Email text | Contains link? | ALL CAPS words | From known sender? | **Spam?** |
| --- | --- | --- | --- | --- |
| "You won $1M! Click here" | Yes | 3 | No | **Yes** |
| "Meeting at 3pm tomorrow" | No | 0 | Yes | **No** |
| "URGENT: Verify account" | Yes | 1 | No | **Yes** |
| "Here's the report you asked for" | Yes | 0 | Yes | **No** |

A traditional programmer looks at this and writes rules ("if ALL CAPS > 2 and not from a known sender, it's spam"). An ML model looks at thousands of these rows and discovers its own rules - including patterns a human might miss, like specific combinations of link placement, word frequency, and sending time.

At its core, ML is about learning patterns from data. A model starts with random guesses, sees examples with correct answers, and gradually adjusts its internal numbers (called weights) until it gets good at predicting. That adjustment process is called training. Using the trained model on new data is called inference. We’ll look at this in the notebook.

## Where ML Shows Up

ML is behind most of the software you use daily. The common thread is problems where writing explicit rules is impractical but examples are plentiful.

| Domain | Example | What ML does |
| --- | --- | --- |
| Recommendations | Spotify Discover Weekly | Predicts songs you'll like from listening patterns |
| Search | Google ranking | Orders billions of pages by relevance to your query |
| Vision | Phone face unlock | Matches your face against a learned representation |
| Language | ChatGPT, Claude | Generates text by predicting likely next words |
| Fraud | Bank transaction alerts | Flags purchases that don't fit your normal behavior |
| Science | AlphaFold | Predicted 3D structure of nearly every known protein |
| Manufacturing | Factory defect detection | Spots bad parts on the production line via camera |

## Types of Machine Learning

 We talk a lot about supervised and unsupervised learning when talking about ML. Supervised learning the most common one, and unsupervised learning is this weird collection of algorithms that can be useful sometimes. Think of these areas of machine learning as algorithms or techniques that solve specific types of problems.

| Type | What it needs | What it does | Examples |
| --- | --- | --- | --- |
| **Supervised** | Labeled data (input + correct answer) | Learns to predict the answer for new inputs | Spam detection, price prediction, image classification |
| **Unsupervised** | Unlabeled data | Finds structure and patterns on its own | Customer segmentation, anomaly detection |
| **Self-supervised** | Raw data (text, images) | Creates its own labels from the data | LLM pretraining ("predict the next word") |
| **Reinforcement** | Environment + rewards | Learns by trial and error | Game AI (AlphaGo), robotics |

Supervised learning is the focus of this course. You have labeled data - inputs paired with correct answers - and the model learns the mapping. It comes in two flavors: classification (predict a category) and regression (predict a number). This thing called “label” is super important. The label is the answer - it’s the data have history on, but in the future we might not have the label. Here's what the data actually looks like:

**Classification example - "will this passenger survive?"**

The columns are the **features** (inputs). The last column is the **label** (what we predict).

| Pclass | Sex | Age | Fare | Survived (label) |
| --- | --- | --- | --- | --- |
| 1 | female | 38 | 71.28 | Yes |
| 3 | male | 22 | 7.25 | No |
| 2 | female | 8 | 25.00 | Yes |
| 1 | male | 54 | 51.86 | No |

**Regression example - "what will this house sell for?"**

| Size (sqft) | Bedrooms | Year Built | Neighborhood | Price (label) |
| --- | --- | --- | --- | --- |
| 1,400 | 3 | 1995 | Suburbs | $285,000 |
| 850 | 1 | 2010 | Downtown | $340,000 |
| 2,200 | 4 | 1980 | Suburbs | $410,000 |

Same idea, different output. Classification predicts a category. Regression predicts a number. The model's job is to find patterns in the features that help predict the label.

Self-supervised learning is worth understanding because it's how modern LLMs work. The model just predicts the next word, over and over, on massive amounts of text. No human labeling needed. We'll work with these models through APIs later in the course.

## Classic ML vs Deep Learning

For most of ML's history, the hard part was feature engineering - manually designing the inputs. You'd spend weeks crafting features like word counts, sender reputation scores, or pixel histograms. The model itself (logistic regression, random forest) was almost an afterthought.

Deep learning changed this for images and text. Instead of hand-crafting features, you feed raw pixels or words into a deep neural network and let it learn its own features. Early layers learn simple patterns (edges, common word pairs), later layers learn complex concepts (faces, sentence meaning).

| Data type | Best approach | Why | When in this course |
| --- | --- | --- | --- |
| Tabular (spreadsheets, databases) | Classic ML - trees, XGBoost | Feature engineering helps, limited data | L3-L7 |
| Images (photos, scans) | Deep learning - CNNs | Learns features from raw pixels | L9-L12 |
| Text (documents, reviews) | Deep learning - Transformers, LLMs | Sequential structure, massive pretraining | L13+ |

We'll learn both approaches. Trees still dominate tabular data, and that's totally fine. The right tool depends on the problem.

## What will we be learning?

### Fastai and pytorch

This course will initially start of using a high level framework called fastai, which is built on top of the worlds most popular framework called pytorch. It’s basically a high level abstraction that takes away some of the details. However, after the first week, we’ll go straight to using pure pytorch, as that is the standard in the field. 

### What about tensorflow?

You might have heard of tensorflow. Well, tensorflow is a framework that used to be very popular, but thesedays its less and less frequently used, and most serious research labs use pytorch. That means you’ll want to avoid videos and courses using tensorflow, even though a lot of overall concepts are still the same. 

### Other things we’ll use

Ultimately, we want to train machine learning models. Pytorch is an amazing framework, but they don’t try to do everything - that means we’ll still use things like pandas in python, we’ll use sci-kit learn here and there for small things.

**pandas** is your go-to for loading and exploring data. Before any model touches your data, you’ll use pandas to read CSVs, check for missing values, look at distributions, and generally understand what you’re working with. If you’ve used it before, great. If not, you’ll pick it up fast, but remember: WE WILL USE AGENTS TO CODE! Don’t stress too much, but try to make sure you understand pandas somewhat.

**scikit-learn** (sklearn) shows up a lot in the first half of the course. It has clean implementations of classic ML algorithms like logistic regression, random forests, and decision trees. It also has handy utilities we’ll use constantly - train/test splitting, accuracy metrics, confusion matrices, data scaling. Think of it as the Swiss army knife of ML. We’ll stop using the algorithms in it, but we’ll keep using it for small utility functions later as well.

**matplotlib** handles all our plotting. Loss curves, data distributions, confusion matrices, feature importance charts - if we’re visualizing something, matplotlib is doing the work.

**numpy** is the math layer underneath everything. Tensors in pytorch are basically numpy arrays with GPU support. You won’t use numpy directly that often, but it’s good to know it’s there.

**XGBoost / LightGBM** are gradient boosting libraries we’ll use in L7 when we look at tree-based models. These are the tools that win most tabular ML competitions on Kaggle, and they’re dead simple to use.

**HuggingFace transformers** shows up later in the course when we work with text and LLMs. It gives you access to thousands of pretrained models with a few lines of code.

Aaaand we might use other stuff - we’ll see! We’ll use whatever seems the most popular, and what makes sense for what we’re trying to do. But the core of the course is pytorch - remember that.

## What We'll Do in the Notebook

The notebook walks through three live demos  - predicting Titanic survival from passenger data (tabular), recognizing pet breeds from photos (images), and analyzing movie review sentiment (text). Three different data types, three different model families, same fundamental pattern: show the model labeled examples, let it find patterns, use those patterns on new data.

After the demos, we'll map out the ML landscape, introduce the core vocabulary (features, labels, training, inference, overfitting), and lay out the course roadmap.

## Terminology

| Term | What it means | Example |
| --- | --- | --- |
| **Features** | The inputs to a model | Age, income, pixel values |
| **Labels** | What we're predicting | Survived/Died, Cat/Dog, $350,000 |
| **Model** | The learned function that maps features to labels | A random forest, a neural network |
| **Training** | Adjusting the model's weights to fit the data | Showing it 10,000 labeled emails |
| **Inference** | Using the trained model on new data | "Is this new email spam?" |
| **Overfitting** | Model memorized training data, fails on new data | 99% on training, 60% on test |

## Getting Started

### Using Google Colab

1. Go to [colab.google.com](https://colab.google.com/)
2. Click **File → Open notebook → GitHub**
3. Authorize GitHub if prompted
4. Paste your repo URL
5. Select the notebook to open
6. Work in Colab, then **File → Save a copy in GitHub** to save back to your fork

Colab gives you a free GPU and a pre-installed Python environment. Most packages are already there. If something is missing: `!pip install fastai transformers xgboost lightgbm`

Colab sessions timeout after ~90 minutes of inactivity. Save your work frequently.

### Working Locally

For local setup, follow the instructions in the repo README. The short version:

**With mamba (recommended):**

```bash
git clone <YOUR_FORK_URL>
cd ua_machine_learning_1
mamba create -n ml-venv python=3.12 -y
mamba activate ml-venv
mamba install pytorch torchvision pytorch-cuda=12.8 -c pytorch -c nvidia -y
mamba install scikit-learn pandas numpy matplotlib seaborn jupyter -y
pip install fastai xgboost lightgbm transformers datasets kaggle
jupyter notebook
```

No GPU? Skip `pytorch-cuda=12.8` and `-c nvidia`. macOS Apple Silicon? `mamba install pytorch torchvision -c pytorch -y`.

### Syncing Updates

When new lessons or fixes are pushed, run `./sync.sh` or manually:

```bash
git fetch upstream
git merge upstream/main
```

### Using Claude Code / Codex

Use AI coding tools to help you learn. Ask them to explain a cell you don't understand, create variations of demos with different data, or debug errors when running notebooks locally. 

## How to learn with agents

As you go through this course, you will do at least these 3 things:
1. Go through my lesson notebooks - before the lesson, and then together with me during the lesson. YOU ALWAYS TRY TO DO IT YOURSELF AHEAD OF THE LESSON! And at that stage, that’s when you should constantly ask your agent - what does this mean? why are we doing this? 
2. Do the homeworks as a way to practice what we did in the lesson notebook - this time however, I only give you and outline of rough instructions. At that stage, you should step by step prompt an agent and try to really understand what you’re doing. Every section of the course have homeworks / projects.

1. The assignments - they are similar to the homeworks. 

When you learn, frequently do things like this:

> Ok, so let me try to explain what a loss function is, correct me if I’m wrong: I think of it as a function that is able to tell me how wrong the prediction was. This is then used in the training loop to calculate gradients, which then is important so that we can update the weights in the right direction. The loss function should help us know in what direction we should move.
> 

Agent: That’s quite right, except you got the XYZ wrong…

Meaning, you should stop yourself and check if you understand things. Use the teaching mode in claude, or tell claude in sessions to act more like a teacher and be critical of the things you’re doing. 

### Create your own content

If you feel like you need something explained from different angles - ask an agent to create a notebook dedicated to the thing you want to understand better. You can essentially build your own course material - but make sure its critical of what it’s doing, and research online to avoid hallucinations. This can be a great way to practice, but also to memorize things. 

### Frequently create notes

Create notes that you can review on the way to school, perhaps while driving and chatting to chatgpt’s advanced voice mode - etc. It’s a great idea to build your own notes as you go, which you could save on notion or similar. In the repository, perhaps create a markdown-file for each lesson with your own notes. That’s how I learned a lot myself. 

## Resources

You’ll find that it’s not that easy to find tutorials on machine learning. 

### Before the lecture

- Fireship - Machine Learning in 100 Seconds (~2 min, fast overview): https://www.youtube.com/watch?v=PeMlgBn-0cs
- Google - Introduction to ML (interactive text): https://developers.google.com/machine-learning/intro-to-ml
- StatQuest - A Gentle Introduction to Machine Learning (12 min, explains supervised vs unsupervised with clear examples): https://www.youtube.com/watch?v=Gv9_4yMHFhI
- IBM Technology - What is Machine Learning? (6 min, good broad overview): https://www.youtube.com/watch?v=9gGnTQTYNaE
- Cassie Kozyrkov - Making Friends with Machine Learning (6-part series from Google's former Chief Decision Scientist, very practical "how to think about ML" framing - a few years old now but the concepts hold up): https://www.youtube.com/watch?v=lYWt-aCnE2U

## The fastai course

The fastai course is a great course to take on the side - it has a bit of a weird structure, and I believe our course is more well structured, but parts of the lessons can be great. Especially now L1 and L2. 

- [fast.ai](http://fast.ai/) Practical Deep Learning - Lesson 1 (top-down intro, similar philosophy): https://course.fast.ai/Lessons/lesson1.html

## What about andrew ng’s machine learning specialization course?

That course is better suited for people who have more time to dive into the math, because it really focuses a lot on the math instead of just doing things practically. The first part of the course can be interesting, but again, I think fastai does a better job, paired with taking this course. Take one week during the summer where you go through it, get the certificate, and put it on linkedin. 

### For the ambitious (optional)

- Kaggle Learn - Intro to Machine Learning (hands-on, short exercises): https://www.kaggle.com/learn/intro-to-machine-learning
- Patrick Loeber - PyTorch Tutorial series (relevant from L3 onwards): https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4