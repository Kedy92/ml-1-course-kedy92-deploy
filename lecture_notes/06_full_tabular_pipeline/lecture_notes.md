# L06: Full Tabular Pipeline

In L4 and L5, we built and understood the MLP - the architecture, the forward pass, backpropagation, the training loop. But we did all of that on the Titanic dataset with 4 clean features and no real mess. Real data isn't like that. It has missing values, categorical strings that need to become numbers, features at wildly different scales, and classes so imbalanced that a model can get 88% accuracy by predicting the same thing every time.

This lesson is about the full pipeline - everything that happens before, during, and after training. If L5 was about understanding the engine, this lesson is about building the whole car: from raw messy CSV to a trained model you can actually evaluate honestly.

Think of it like building a web app. You wouldn't ship a FastAPI endpoint without input validation, error handling, and proper testing. The ML pipeline is the same discipline applied to a different domain. And just like web development, most of the work isn't in the core logic - it's in everything around it.

## The Pipeline

The lesson follows a complete ML project from raw CSV to evaluated model. Every concept lives inside one of these steps - remember, the actual steps are generally very similar, but the details might change depending on our dataset, what type of problem we’re solving, but I think it’s very healthy to go through a proper pipeline. It will help us get better at pytorch, and overall this will be a far more practical lesson in terms of actually creating a model, a realistic workflow you’ll do many times from now on.

### Step 1: Explore the data

Before any model code, understand what you're working with. This is where you catch **data leakage** (features that wouldn't exist at prediction time), **class imbalance** (88% of one class makes accuracy meaningless), missing values, and **sentinel values** (999 pretending to be a real number). Skipping this step is how you train a model that looks great in a notebook but fails in production.

### Step 2: Preprocess

Neural networks only understand numbers. Categorical strings get **label encoded** into integer IDs, which become lookup keys for **embeddings** (learned in the model). Numerical features get **standard scaled** so no feature dominates the gradients. We split into train/validation first, then fit the scaler on training data only - fitting on the full dataset before splitting is a subtle form of data leakage.

### Step 3: Bridge to PyTorch

Pandas DataFrames become PyTorch tensors via a **Dataset** (storage) and **DataLoader** (batching and shuffling). This is the standard PyTorch pattern you'll reuse in every project going forward - tabular, images, text.

### Step 4: Build the model

The big new concept here is **embeddings** - each categorical column gets a small learnable lookup table. The `job` column with 12 categories gets a 12x6 table where each row is a learned vector. These get concatenated with the scaled numerical features and fed through the same MLP architecture from L5.

We also switch from `BCELoss` (L5) to `BCEWithLogitsLoss` - same idea, but it applies sigmoid internally and is numerically more stable.

### Step 5: Train

Same training loop from L5 - forward, loss, backward, update - but now running over batches from a DataLoader. **Class imbalance** gets handled here with `pos_weight` in the loss function, making the model care more about the minority class instead of always predicting "no."

### Step 6: Tune hyperparameters

Neuron count, learning rate, dropout, weight decay - these are **hyperparameters** the model can't learn on its own. We experiment systematically: no regularization (overfits fast), heavy regularization (underfits), balanced (sweet spot). Then sweep each parameter individually. We also introduce **early stopping** - stop training when validation loss stops improving.

### Step 7: Evaluate

This is where most beginners stop too early. Accuracy alone is misleading with imbalanced data - a model predicting "no" for everything gets 88%. We need the full picture: **confusion matrix** (where exactly does the model get it right and wrong), **precision** (don't cry wolf), **recall** (don't miss real positives), **F1** (balances both), and **AUC-ROC** (overall ranking quality using raw probabilities, not thresholds).

Which metric matters most depends on the business. Missing a subscriber vs wasting a phone call are different costs - the model doesn't know your priorities, that decision is yours.

## Terminology

| Term | What it means | Where we see it |
| --- | --- | --- |
| **Feature scaling** | Putting all numerical features on the same scale | StandardScaler on numerical columns |
| **Label encoding** | Converting categories to integer IDs | "admin." becomes 0, "blue-collar" becomes 1 |
| **Embedding** | Learnable lookup table for categorical features | nn.Embedding(12, 6) for the job column |
| **Train/validation split** | Splitting data so model trains on one part, gets tested on another | 80/20 split with stratification |
| **Class imbalance** | When one class dominates the dataset | 88% "no" vs 12% "yes" |
| **pos_weight** | Loss multiplier making the model care more about the minority class | BCEWithLogitsLoss(pos_weight=7.88) |
| **Confusion matrix** | Grid showing predicted vs actual for each class | TP, FP, TN, FN counts |
| **Precision** | Of everything predicted positive, how many were correct | "Don't cry wolf" metric |
| **Recall** | Of everything actually positive, how many did you catch | "Don't miss real wolves" metric |
| **F1 score** | Harmonic mean of precision and recall | Single number balancing both |
| **AUC-ROC** | Probability model ranks a random positive above a random negative | 0.5 = random, 1.0 = perfect |
| **Cross-validation** | Multiple train/val splits averaged for robust evaluation | k-fold CV |
| **Hyperparameter** | Setting you choose before training (not learned) | Learning rate, dropout, neurons, weight decay |
| **Data leakage** | When training data contains info that wouldn't exist at prediction time | Call duration in bank marketing data |

## Connection to L5 and L7

**From L5:** We opened the MLP's black box - tracing the forward pass, understanding backpropagation, dissecting the training loop, and comparing optimizers. All of those pieces reappear here, but now wrapped in a real end-to-end pipeline with proper data prep and evaluation. The training loop is the same one from L4-L5, just applied to messier data with more thoughtful setup around it.

**To L7:** We've now trained an MLP on tabular data with a solid pipeline. But is a neural network the best tool for tabular data? Lesson 7 introduces decision trees, random forests, and gradient boosting - and honestly, trees often win on tabular. We'll compare them head-to-head with the MLP approach on a new dataset. The pipeline skills from L6 (data exploration, preprocessing, evaluation) carry over directly.

## Resources

Once again, there isn't much content related to tabular data with neural networks, perhaps some payed courses might have that, but not on youtube or articles, thus the lesson will be extra important, and your own prompting doing the homework will be even more important.

- I’d say keep watching this series - but remember, we’ll focus on tabular data here and not images https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
- PyTorch Dataset & DataLoaders tutorial (official docs, walks through building custom datasets and batching): https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- PyTorch classification from scratch (step-by-step dense neural network for classification): https://www.slingacademy.com/article/pytorch-classification-from-scratch-building-a-dense-neural-network/
- PyTorch for tabular data with categorical embeddings (the embedding approach we use in the notebook): https://yashuseth.wordpress.com/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/
- Google ML Crash Course - Classification metrics (interactive explanation of accuracy, precision, recall): https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall

### Quick videos

- DataLoaders explained (~10 min, visual walkthrough of PyTorch DataLoaders): https://www.youtube.com/watch?v=PXOzkkB5eH0
- NYC taxi fares with PyTorch (practical tabular example, end-to-end): https://francescopochetti.com/pytorch-for-tabular-data-predicting-nyc-taxi-fares/

### Additional reading

- Machine Learning Mastery - What is a Confusion Matrix (article, walks through the matrix with examples): https://machinelearningmastery.com/confusion-matrix-machine-learning/
- PyTorch Tabular Binary Classification (simpler version of what we do in the notebook - same pattern with Dataset/DataLoader/BCEWithLogitsLoss but on a small all-numerical dataset, no embeddings. Good warmup if you want to see the basic pipeline before tackling the full notebook): https://medium.com/data-science/pytorch-tabular-binary-classification-a0368da5bb89