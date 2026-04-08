# Assignment 2: Tabular ML Pipeline

> **Note:** This is an example assignment from "Practical Machine Learning for Programmers."
> If you are completing this as part of a course, follow the submission and grading instructions
> from your official GitHub Classroom invitation.

**Covers:** Chapters 5-7

## What are we doing?

Pick a tabular dataset not used in the lesson notebooks, build two models (a PyTorch MLP and a tree-based model), compare them, and deploy them behind a single API.

This is the full workflow from Chapters 5-7 applied to a new problem. You'll go through the entire pipeline: explore messy data, preprocess it, train a neural network with embeddings, train a gradient boosting model, evaluate both properly, and ship them as a working API. The point isn't to get the highest accuracy - it's to demonstrate that you can take a raw dataset and build something real with it.

## The assignment

### 1. Pick a dataset

Choose a tabular dataset that:

- Was **not** used in any lesson notebook (not Titanic, not Bank Marketing, not Bulldozers)
- Has a **mix of categorical and numerical features** (so you practice embeddings)
- Has at least 1,000 rows
- Is a **classification** problem (binary or multi-class)

Some options to get you started (or find your own):

| Dataset | Rows | Task | Source |
| --- | --- | --- | --- |
| Adult Income | 48K | Binary (>50K income?) | UCI ML Repository |
| Telco Customer Churn | 7K | Binary (churned?) | Kaggle |
| Heart Disease | 900+ | Binary (disease?) | UCI ML Repository |
| Spaceship Titanic | 8.6K | Binary (transported?) | Kaggle |
| Credit Card Default | 30K | Binary (default?) | UCI ML Repository |
| Mushroom Classification | 8K | Binary (poisonous?) | UCI ML Repository |

You can use any tabular classification dataset your agent confirms would work. Larger is generally better.

### 2. Build the MLP pipeline (PyTorch)

Follow the same pipeline from Chapter 6:

- **Explore** the data - check for missing values, class balance, data leakage, feature distributions
- **Preprocess** - label encode categoricals, standard scale numericals, train/val split with stratification
- **Build a Dataset and DataLoader** - custom PyTorch Dataset class, batched DataLoader
- **Build an MLP with embeddings** - embedding layers for categoricals, concatenated with numericals, fed through linear layers with ReLU
- **Train** - training loop with validation tracking. Handle class imbalance with `pos_weight` if needed
- **Tune** - try different hyperparameter settings (e.g. network size, dropout, learning rate). Show the results
- **Evaluate** - confusion matrix, precision, recall, F1, AUC-ROC

### 3. Build a tree model

Using the same dataset:

- Train a **gradient boosting model** (XGBoost, LightGBM, or CatBoost) with early stopping, use a random forest as a baseline to compare against
- **Feature importance** - show which features matter most
- **SHAP** - generate a summary plot (beeswarm or bar) showing how features affect predictions
- Write your own reflection on the results - don't just AI-generate a summary

### 4. Compare the models

Show a comparison table with metrics for both models. Something like:

```
Model                  Accuracy   F1      AUC-ROC
MLP (best config)      0.82       0.74    0.88
LightGBM               0.84       0.77    0.91
```

No essay needed. The numbers and plots speak for themselves.

### 5. Deploy both models

Build a **single FastAPI app** that serves predictions from both models behind different endpoints:

```
POST /predict/mlp      -> {"prediction": "yes", "probability": 0.73}
POST /predict/tree     -> {"prediction": "yes", "probability": 0.81}
```

Both endpoints accept the same JSON input (the feature values for one sample) and return a prediction with probability.

It should:

- Accept JSON input with feature values
- Return the predicted class and probability
- Run locally with `docker-compose up` (or equivalent)
- Include a README explaining how to run it and what input format to use

### 6. Share your results

Document your work:

1. **Your dataset** - name and link to where you got it
2. **Your comparison table** - the actual output from your notebook showing metrics for both models
3. **Your SHAP summary plot** (from the tree model)
4. **A working demo** - either a Streamlit app, a short video walkthrough, or a link to the running API

### 7. BONUS (Optional)

If you end up with the time, and you can find a usecase given what your models actually predict, you could vibe-engineer a basic application that delivers some type of business value. For example, an application that calculates video game sales estimates based on ML-models, or a dashboard that predicts approval probabilities based on tabular data. Perhaps any of your previous fullstack projects could be relevant?

## How to work

Use Claude Code, Cursor, or any AI coding tool. It's encouraged. But don't just accept what the agent gives you. The goal is understanding.

Some specific things to watch out for:

- **Preprocessing matters more than you think.** If your MLP performs badly, check your preprocessing first. Did you scale the numericals? Did you encode the categoricals correctly? Did you split before fitting the scaler?
- You should primarily use PyTorch and only bring in external dependencies when it makes sense
- **Early stopping for trees.** Don't train gradient boosting without early stopping. Set `n_estimators` high and let early stopping decide when to stop.
- **SHAP is for the tree model, not the MLP.** SHAP has a fast exact algorithm for trees (TreeExplainer) but is slow and approximate for neural networks. Run SHAP on your gradient boosting model only. Use a subset of your validation data (e.g. 500-1000 samples) since even TreeExplainer can be slow on large datasets.

## What you should produce

- **A notebook** showing the full pipeline: data exploration, MLP training with embeddings, tree model training, SHAP analysis, comparison table, evaluation metrics for both models
- **A working deployment** (FastAPI app with both model endpoints) with a README explaining how to run it and the expected input format
- **Your saved models** - the PyTorch model state dict and the tree model pickle/joblib. If too large for git, include download instructions.

## Tips

- **Start with the data.** Spend real time understanding your dataset before writing any model code. The exploration phase saves you hours of debugging later.
- **Get the MLP working first.** The tree model is easier to get running, so save it for after. The MLP pipeline (Dataset, DataLoader, embeddings, training loop) is where the learning happens.
- **Test your deployment early.** Don't wait until the last day. Get a basic FastAPI endpoint working with dummy predictions first, then swap in your real models.
- **The SHAP plot is important.** It's one of the most valuable skills from Chapter 7 - being able to explain what your model learned. Don't skip it.
- **Class imbalance is common.** Most real datasets are imbalanced. Check your target distribution early and decide how to handle it (pos_weight, class_weight, or just being aware of it during evaluation).
- Feel free to do either regression or classification - it's up to you!
