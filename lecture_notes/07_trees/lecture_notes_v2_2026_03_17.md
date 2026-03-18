# L07: Trees - Random Forests & Gradient Boosting

In Lesson 6, we built a full pipeline for tabular data using an MLP with embeddings. We dealt with missing values, encoded categories, scaled features, handled class imbalance, and evaluated with proper metrics. The MLP worked, but here's the honest truth: for tabular data, tree-based models often win. They're faster to train, easier to interpret, need less preprocessing, and frequently deliver equal or better accuracy. This lesson is about giving you the right tool for the job.

We'll build decision trees, random forests, and gradient boosting models (XGBoost, LightGBM, CatBoost) on a real dataset and compare them head-to-head with the MLP approach from L6. The goal isn't to pick a favorite - it's to understand when each approach makes sense. By the end, you'll have a practical framework: try trees first on tabular, use neural nets for images and text, and always compare.

## Key Concepts

**Decision Trees** - the building block of everything in this lesson. A decision tree makes predictions by asking a series of yes/no questions. "Is the year made before 2000?" If yes, go left. If no, go right. Keep splitting until you reach a leaf, which holds the prediction (the average value of all training samples that ended up there).

The tree algorithm picks splits automatically by testing every feature at every possible threshold, choosing the one that creates the most homogeneous groups. For regression, "homogeneous" means low variance (prices within each group are similar). For classification, it means low Gini impurity (most samples in each group belong to the same class).

Pros: interpretable (you can read the rules), no preprocessing needed (trees don't care about scale), handles missing values. Cons: a single deep tree overfits badly - it memorizes the training data by creating leaves with just a few samples each.

**Random Forests** - the fix for overfitting. Train many decision trees (typically 100+), each on a random subset of the data (bootstrap sampling) using a random subset of features at each split. Then average their predictions.

Why does averaging work? Each tree sees different data and different features, so they make different mistakes. When you average many uncorrelated errors, they tend to cancel out. The result is a model that's much more stable and accurate than any single tree. It's the same principle as asking 100 people to estimate the weight of a cow - the average is usually better than any individual guess.

**Bagging (Bootstrap Aggregating)** - the technique behind random forests. Draw N samples with replacement from N training samples. Some samples appear multiple times, roughly 37% don't appear at all. Train a tree on each bootstrap sample. The samples that didn't appear become the "out-of-bag" (OOB) set - a free validation set for each tree.

**Out-of-Bag Error** - for each training sample, some trees never saw it during training. We can collect predictions from only those trees and get a validation-like score without setting aside any data. Useful when data is limited. For time-series data, use a proper temporal split instead.

**Feature Importance** - random forests track how much each feature contributes to reducing prediction error across all trees. This tells you which features matter most. In the notebook, we'll see that a handful of features dominate while dozens contribute almost nothing - a common pattern that lets us simplify the model.

**Prediction Confidence** - since a random forest is 100 trees each giving their own prediction, the spread (standard deviation) of those predictions tells you how confident the model is. Low spread means the trees agree - high confidence. High spread means they disagree - the model is uncertain. This is practically useful: in production, you can route uncertain predictions to human review.

**Gradient Boosting** - a different ensemble strategy. Instead of training trees independently and averaging (random forest), train trees sequentially. Each new tree focuses specifically on the errors of all previous trees combined. Tree 1 makes predictions. Tree 2 predicts the residuals (mistakes) of tree 1. Tree 3 predicts the residuals of trees 1+2 combined. Stack enough trees and the ensemble gets very accurate.

The tradeoff: gradient boosting is more powerful but also more fragile. It can overfit if you add too many trees or use too high a learning rate. That's why early stopping is essential - monitor validation performance and stop adding trees when it stops improving.

**XGBoost, LightGBM, CatBoost** - three popular implementations of gradient boosting. XGBoost is the battle-tested standard. LightGBM is faster (great for quick experiments). CatBoost handles categorical features natively. In practice, the differences in accuracy are usually small - good feature engineering matters more than library choice.

**SHAP (SHapley Additive exPlanations)** - goes beyond feature importance. Instead of "which features matter globally?", SHAP answers "why did the model make this specific prediction?" For each prediction, SHAP shows how each feature pushed the prediction up or down from the average. A waterfall plot shows one prediction decomposed. A beeswarm plot shows patterns across all predictions.

**When to Use What** - a practical decision framework:
- Tabular data: try trees first. They're fast, need minimal preprocessing, and often best
- Images/text/audio: neural nets. Trees can't handle spatial or sequential structure
- Small data: trees. They need fewer samples to learn useful patterns
- Interpretability matters: trees or SHAP on top of gradient boosting
- Always worth trying both and comparing honestly

**Tree Extrapolation Limitation** - trees can only predict values within the range they saw during training. If training prices go up to $100k, the tree can never predict $150k. Neural nets don't have this constraint. Keep this in mind with time-series or trend data.

## Terminology

| Term | What it means | Where we see it |
| --- | --- | --- |
| **Decision tree** | Model that predicts by following a chain of yes/no splits | Visualized tree with nodes and leaves |
| **Gini impurity** | How mixed a group is after splitting (classification) | Split quality scoring |
| **Random forest** | Ensemble of trees trained on random data/feature subsets | 100 trees averaged together |
| **Bagging** | Bootstrap aggregating - train on random samples with replacement | Each tree sees ~63% of data |
| **Out-of-bag (OOB)** | Samples a tree never trained on, used as free validation | ~37% of data per tree |
| **Ensemble** | Combining multiple models to get better predictions | Random forest, gradient boosting |
| **Feature importance** | How much each feature reduces prediction error across all trees | Bar chart of top features |
| **Gradient boosting** | Trees trained sequentially, each correcting previous errors | XGBoost, LightGBM, CatBoost |
| **Early stopping** | Stop training when validation performance stops improving | Prevents gradient boosting from overfitting |
| **Learning rate (boosting)** | How much each new tree's correction counts | Lower = more trees needed, but more stable |
| **SHAP values** | Per-feature contribution to a specific prediction | Waterfall and beeswarm plots |
| **Extrapolation** | Predicting beyond the range seen in training | Trees can't do this, neural nets can |
| **Log transform** | Taking log of skewed target to normalize the distribution | log(price) makes RMSE work better |

## Connection to L6 and L8

**From L6:** The pipeline skills transfer directly - data exploration, preprocessing, train/val splitting, evaluation metrics. The difference is that trees need much less preprocessing (no normalization, no embeddings, handles missing values). The confusion matrix, precision, recall, and F1 concepts from L6 apply exactly the same way to tree classifiers. We also rebuild the MLP from L6 for a fair head-to-head comparison.

**To L8:** L8 is a project assistance lesson with no new content. You consolidate everything from L1-L7 - both the neural net pipeline (L5-L6) and the tree-based approach (L7). The goal is solid tabular ML fluency before switching to a fundamentally different data domain: images. The pipeline patterns, evaluation discipline, and "right tool for the job" thinking from these lessons carry forward into everything that follows.

## Resources

### Course video

TBA

### Before the lesson

- StatQuest - Decision and Classification Trees, Clearly Explained (18 min, excellent visual walkthrough of how trees split data using Gini impurity, a must-watch before the lesson): https://www.youtube.com/watch?v=_L39rN6gz7Y
- StatQuest - Random Forests Part 1: Building, Using and Evaluating (10 min, builds from decision trees to the full random forest concept): https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
- StatQuest - Regression Trees, Clearly Explained (9 min, how trees handle continuous targets instead of categories): https://www.youtube.com/watch?v=g9c66TUylZ4

### Deep dives

- Random Forests (fastai lesson, ~90 min, Jeremy Howard walks through random forests on the bulldozer dataset - the same dataset we use in the notebook): https://www.youtube.com/watch?v=AdhG64NF76E&list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU&index=7
- Gradient Boosted Trees (StatQuest, 15 min, explains the sequential learning process clearly - watch after understanding decision trees and random forests): https://www.youtube.com/watch?v=3CC4N4z3GJc
- Gradient Boosting for classification (StatQuest, 11 min, extends the regression explanation to classification tasks): https://www.youtube.com/watch?v=jxuNLH5dXCs

### Model interpretation

- SHAP - Model Interpretation (22 min, visual explanation of SHAP values and how to read the plots we use in the notebook): https://www.youtube.com/watch?v=MQ6fFDwjuco
- Model Interpretable vs Explainable (short, clarifies the difference between models you can read directly vs models you need tools to explain): https://www.youtube.com/watch?v=VY7SCl_DFho

### Documentation

- scikit-learn Decision Trees (official docs, covers both classification and regression trees with API reference): https://scikit-learn.org/stable/modules/tree.html
- scikit-learn Random Forests (official docs, ensemble methods including random forests and gradient boosting): https://scikit-learn.org/stable/modules/ensemble.html
- XGBoost documentation (getting started guide and parameter tuning): https://xgboost.readthedocs.io/en/latest/
- LightGBM documentation (quick start and key differences from XGBoost): https://lightgbm.readthedocs.io/en/latest/
- SHAP documentation (Python library for model interpretation, includes tutorial notebooks): https://shap.readthedocs.io/en/latest/

### Additional videos

- Gradient Boosted Trees overview (StatQuest, 10 min, covers the core idea of gradient boosting at a higher level): https://www.youtube.com/watch?v=PxgVFp5a0E4
