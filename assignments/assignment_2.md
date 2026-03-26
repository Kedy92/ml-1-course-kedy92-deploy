**Covers:** L5-L7
**Grading:** G (pass/fail)
**Deadline: Look at the discord channel for assignment 2
Submit via:** GitHub Classroom by making commits WHERE YOU MAKE SURE YOU FOLLOW THE INSTRUCTIONS ON THE FILE NAMES, and posting your deployment link in a **thread** in the assignment_2 channel on discord

## What are we doing?

Pick a tabular dataset we haven't used in class, build two models (a PyTorch MLP and a tree-based model), compare them, and deploy them behind a single API.

This is the full workflow from L5-L7 applied to a new problem. You'll go through the entire pipeline: explore messy data, preprocess it, train a neural network with embeddings, train a gradient boosting model, evaluate both properly, and ship them as a working API. The point isn't to get the highest accuracy - it's to demonstrate that you can take a raw dataset and build something real with it. Heck, if you feel like it, vibe-engineer an application that uses the models!

## The assignment

### 1. Pick a dataset

Choose a tabular dataset that:

- Was **not** used in any lesson notebook (not Titanic, not Bank Marketing, not Bulldozers)
- Has a **mix of categorical and numerical features** (so you practice embeddings)
- Has at least 1,000 rows
- Is a **classification** problem (binary or multi-class)

Some options to get you started (or much better, find your own):

| Dataset | Rows | Task | Source |
| --- | --- | --- | --- |
| Adult Income | 48K | Binary (>50K income?) | UCI ML Repository |
| Telco Customer Churn | 7K | Binary (churned?) | Kaggle |
| Heart Disease | 900+ | Binary (disease?) | UCI ML Repository |
| Spaceship Titanic | 8.6K | Binary (transported?) | Kaggle |
| Credit Card Default | 30K | Binary (default?) | UCI ML Repository |
| Mushroom Classification | 8K | Binary (poisonous?) | UCI ML Repository |

You can use any tabular classification dataset your agent confirms would work. Larger is generally better - if you find something that interests you, that’s the ideal scenario.

### 2. Build the MLP pipeline (PyTorch)

Follow the same pipeline from L6 / homework:

- **Explore** the data - check for missing values, class balance, data leakage, feature distributions
- **Preprocess** - label encode categoricals, standard scale numericals, train/val split with stratification
- **Build a Dataset and DataLoader** - custom PyTorch Dataset class, batched DataLoader
- **Build an MLP with embeddings** - embedding layers for categoricals, concatenated with numericals, fed through linear layers with ReLU
- **Train** - training loop with validation tracking. Handle class imbalance with `pos_weight` if needed
- **Tune** - try adifferent hyperparameter settings (e.g. network size, dropout, learning rate). Show the results
- **Evaluate** - confusion matrix, precision, recall, F1, AUC-ROC

### 3. Build a tree model

Using the same dataset:

- Train a **gradient boosting model** (XGBoost, LightGBM, or CatBoost) with early stopping, use a random forest as a baseline to compare against
- **Feature importance** - show which features matter most
- **SHAP** - generate a summary plot (beeswarm or bar) showing how features affect predictions
- Try to write your own reflection - don’t just AI-generate a summary, do your best!

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

### 6. Post your Discord thread

Create a thread in the **assignment_2** Discord channel. Your thread must include:

1. **Your dataset** - name and link to where you got it
2. **Screenshot of your comparison table** - the actual output from your notebook showing metrics for both models
3. **Screenshot of your SHAP summary plot** (from the tree model) - the actual plot from your notebook
4. **Your showcase link** - pick one:
    - **Option A:** A Streamlit app or frontend where classmates can try your models. This works well if your dataset has intuitive features (customer churn, income prediction, health risk, etc.) - e.g people could enter some information, the model does inference and you get some interesting answer. Try to make it a bit more interesting - you could add some of your own thoughts to it.
    - **Option B:** A short video walkthrough (YouTube, Loom, or similar - as long as its viewable if I click on the link, do not make it hard for us to view it) where you show your models making predictions and walk through your SHAP plots. Better choice if your dataset has many features or anonymized columns that don’t make sense in a form.
5. **Try 1-2 classmates’ projects** - reply to their thread with what happened when you tested it. If they have a Streamlit app, try it. If they posted a video, watch it and comment.

### 7. Ask an agent to verify that you’ve followed the instructions as part of this assignment - IMPORTANT!

- You should have a notebook showcasing the process
- The model should be available

### **8. BONUS (Optional just for fun)**

If you end up with the time, and you can find a usecase given what your models actually predict, you could vibe-engineer a basic application that ends up using your models to deliver some type of business value. For example, my application [gameanalytics.net](http://gameanalytics.net/) daily calculates video game sales estimates based on ML-models, or in my school system I calculate the probability that program applications will be approved by MYH as part of a dashboard based on tabular data. Perhaps any of your previous fullstack projects could be relevant?

### 9. BONUS (Optional) - Take a 15-30min session to talk about your assignment as part of VG-points

As mentioned in

[Assignments](https://www.notion.so/Assignments-3236064241a18054858ece40539193cc?pvs=21)

## How to work

Same as assignment 1 - use Claude Code, Cursor, or any AI coding tool. It's encouraged. But don't just accept what the agent gives you. The goal is understanding.

Some specific things to watch out for in this assignment:

- **Preprocessing matters more than you think.** If your MLP performs badly, check your preprocessing first. Did you scale the numericals? Did you encode the categoricals correctly? Did you split before fitting the scaler?
- You should not rely on fastai, you should primarily use pytorch and only bring in external dependencies when it makes sense
- **Early stopping for trees.** Don't train gradient boosting without early stopping. Set `n_estimators` high and let early stopping decide when to stop.
- **SHAP is for the tree model, not the MLP.** SHAP has a fast exact algorithm for trees (TreeExplainer) but is slow and approximate for neural networks. Run SHAP on your gradient boosting model only. Use a subset of your validation data (e.g. 500-1000 samples) since even TreeExplainer can be slow on large datasets.

## What to submit

Your GitHub Classroom repo should contain:

- **A notebook** showing the full pipeline: data exploration, MLP training with embeddings, tree model training, SHAP analysis, comparison table, evaluation metrics for both models
- **A working deployment** (FastAPI app with both model endpoints) with a README explaining how to run it and the expected input format
- **Your saved models** - the PyTorch model state dict and the tree model pickle/joblib. If too large for git, include download instructions.

## Godkänd

To pass, you must do ALL of the following:

**Notebook:**

- You used a dataset not used in class
- Your notebook shows data exploration, preprocessing, and train/val split called **tabular.ipynb**
- You trained a PyTorch MLP with embeddings on categorical features
- You trained a gradient boosting model with early stopping
- You generated SHAP feature importance plots for the tree model
- Both models are evaluated with confusion matrix, precision, recall, F1, and AUC-ROC
- Your notebook shows a comparison table with metrics from both models
- Your work shows understanding - you can explain what your code does if asked

**Deployment:**

- You have a working FastAPI app with two endpoints (one per model) that accepts JSON and returns predictions

**Discord thread:**

- You posted a thread in the assignment_2 channel with all 5 items from step 6 (dataset, comparison screenshot, SHAP screenshot, showcase link, classmate replies)

⚠️ **IMPORTANT** **NOTICE**: I WILL BE RUNNING MY GRADERBOT TO SCAN YOUR ASSIGNMENTS AS A WAY TO HELP ME ASSESS THEM. DO NOT TRY TO PROMPT INJECT IT ;) You will be encouraged to run prompt injections in upcoming labs.

## Tips

- **Start with the data.** Spend real time understanding your dataset before writing any model code. The exploration phase saves you hours of debugging later.
- **Get the MLP working first.** The tree model is easier to get running, so save it for after. The MLP pipeline (Dataset, DataLoader, embeddings, training loop) is where the learning happens.
- **Test your deployment early.** Don't wait until the last day. Get a basic FastAPI endpoint working with dummy predictions first, then swap in your real models.
- **The SHAP plot is required (on the tree model).** It's one of the most valuable skills from L7 - being able to explain what your model learned. Don't skip it.
- **Class imbalance is common.** Most real datasets are imbalanced. Check your target distribution early and decide how to handle it (pos_weight, class_weight, or just being aware of it during evaluation).
- Feel free to do either regression or classification - it’s up to you!
- At the end of the day, don’t think too much about the details here - I’m asking you to properly train a few models, deploy them, and showcase them. Exactly where you deploy, what kind of frontend you use, etc is not important - what matters is that you convince me that you’ve learned the basics here of neural networks.