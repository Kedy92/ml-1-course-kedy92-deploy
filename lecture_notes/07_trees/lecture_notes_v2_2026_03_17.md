# L07: Trees - Random Forests & Gradient Boosting

In L5 and L6, we built neural networks for tabular data - first understanding the mechanics, then building a full pipeline with embeddings, preprocessing, and evaluation. The MLP worked. But here's the honest truth: for tabular data, tree-based models often win. They're faster to train, easier to interpret, need less preprocessing, and frequently deliver equal or better accuracy.

This lesson introduces a completely different family of algorithms. All of these algorithms work for both regression (predicting a number) and classification (predicting a category) - the core logic is the same, only the scoring function changes. The notebook focuses on regression (predicting bulldozer prices), but everything you learn applies directly to classification too.

We start with **decision trees** - the simplest tree model and the foundation everything else builds on. A single decision tree is easy to understand but overfits badly. So we fix that with **random forests**, which combine hundreds of trees to cancel out individual mistakes. Then we move to **gradient boosting**, where trees are built sequentially, each one correcting the errors of the previous ones. Gradient boosting (XGBoost, LightGBM, CatBoost) is what dominates Kaggle competitions and real-world tabular ML - it’s generally the best-performing approach for structured data. But again, neural networks could actually still be better - so this is just another toolkit in our toolbox, and it’s a really lovely one, but most machine learning engineers would then end up trying both to see which one is performing the best.

## Key Concepts

### Decision Trees - the foundation

Everything in this lesson builds on decision trees, so it's worth understanding them well before moving on.

A decision tree makes predictions by asking a series of yes/no questions. "Is the year made before 2000?" If yes, go left. If no, go right. Keep splitting until you reach a leaf. For regression (predicting a number like price), each leaf holds the average value of the training samples that ended up there. For classification (predicting a category), each leaf holds the majority class. The tree algorithm picks splits automatically by testing every feature at every possible threshold, choosing the one that creates the most homogeneous groups - measured by variance for regression, or Gini impurity for classification.

The appeal of decision trees is that you can read them. You can look at the splits and understand exactly why the model made a prediction. They also need almost no preprocessing - no normalization, no encoding, they handle missing values naturally. But a single deep tree has a serious problem: it overfits. Given enough depth, a tree will create a leaf for every single training example, memorizing the data perfectly while learning nothing that generalizes to new data.

### Random Forests - fixing overfitting with ensembles

The solution to a single tree's overfitting problem is to build many trees and average their predictions. That's a random forest.

Each tree is trained on a slightly different version of the data (a **bootstrap sample** - random sampling with replacement, so each tree sees about 63% of the data). At each split, the tree only considers a random subset of features. This randomness means each tree makes different mistakes. When you average 100 trees that make uncorrelated errors, those errors tend to cancel out - the same principle as asking 100 people to estimate something and taking the average.

Random forests also give you useful things for free. **Feature importance** tells you which features contribute most to predictions across all trees - typically a handful dominate while most contribute almost nothing. **Out-of-bag (OOB) error** gives you a validation score without setting aside any data, since each tree has ~37% of samples it never trained on. And the spread of predictions across the 100 trees gives you **prediction confidence** - if all trees agree, you can trust the prediction; if they disagree, route it to human review.

### Gradient Boosting - learning from mistakes

Random forests build trees independently and average them. Gradient boosting takes a different approach: build trees sequentially, where each new tree specifically targets the errors the previous trees made.

It starts with a simple prediction (the average value). Then it calculates the residuals - how far off each prediction is from the truth. It trains a small tree to predict those residuals. It adds that tree's predictions to the running total, scaled down by a **learning rate** (typically 0.05-0.1) so each tree takes a small step rather than overcorrecting. Then it calculates new residuals and trains another tree. Repeat hundreds or thousands of times, each tree nudging the predictions a little closer to the truth.

The tradeoff: gradient boosting is more powerful than random forests but also more fragile. Too many trees or too high a learning rate and it overfits. That's why **early stopping** is essential - monitor validation performance and stop adding trees when it stops improving. This is probably the single most important practical tip for gradient boosting.

### The libraries: XGBoost, LightGBM, CatBoost

Three popular implementations of gradient boosting. **XGBoost** is the battle-tested standard that dominated Kaggle for years. **LightGBM** is faster (great for quick experiments and large datasets). **CatBoost** handles categorical features natively without manual encoding. In practice, the accuracy differences between them are usually tiny - good feature engineering matters far more than library choice. The notebook compares all three side by side.

### Interpretability with SHAP

One of the biggest advantages of tree-based models is that we can look inside them. **Feature importance** tells us which features matter globally. **SHAP (SHapley Additive exPlanations)** goes further and answers "why did the model make *this specific prediction*?" For each prediction, SHAP shows how each feature pushed the result up or down from the average. A waterfall plot decomposes one prediction. A beeswarm plot shows patterns across all predictions. This kind of interpretability is much harder to get from neural networks.

### When to use what

- **Tabular data**: try trees first. They're fast, need minimal preprocessing, and often best
- **Images/text/audio**: neural nets. Trees can't handle spatial or sequential structure
- **Small data**: trees. They need fewer samples to learn useful patterns
- **Interpretability matters**: trees, or SHAP on top of gradient boosting
- Always worth trying both and comparing honestly

One important limitation to keep in mind: **trees can't extrapolate**. A tree's prediction is always within the range of values it saw during training. If training prices go up to $100k, the tree can never predict $150k. Neural nets don't have this constraint. This matters with time-series or trend data.

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

This lesson by Jeremy Howard in the fastai course is a great place to start with tree models:

- The first lecture which you can start watching from 1:26:40 https://www.youtube.com/watch?v=_rXzeWq4C6w&list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU&index=5
- And the continuation of that lecture all the way up until 52 minutes https://www.youtube.com/watch?v=AdhG64NF76E&list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU&index=8

They are based on a lecture notebook which is included in your repository in the fastai_course_notebooks just for convenience (*07-how-random-forests-really-work.ipynb*) and *09_tabular.ipynb*, which is from his book “fastbook” which is more theory heavy but is worth reading. Note that the 07 notebook uses the Titanic dataset for classification, which is a nice complement to our lesson notebook that focuses on regression (bulldozer prices). Once you’ve gone through these, the UA course lecture notebook will be better since its quite similar, but is more realistic, and streamlined - it mixes theory, but it also goes more into depth on gradient boosted trees, as they are generally more favored in real projects.

### Other videos that can be great to get different perspectives

- StatQuest - Decision and Classification Trees, Clearly Explained (18 min, excellent visual walkthrough of how trees split data using Gini impurity, a must-watch before the lesson): https://www.youtube.com/watch?v=_L39rN6gz7Y → He has more videos on decision trees, you don’t have to watch them, but at some point you may want to deepdive.
- StatQuest - Random Forests Part 1: Building, Using and Evaluating (10 min, builds from decision trees to the full random forest concept): https://www.youtube.com/watch?v=J4Wdy0Wc_xQ → Decision trees are the foundation of tree based models, but random forests is a clear usable algorithm that actually uses decision trees (think of it like creating massive amounts of decision trees, we call those random forests)
- StatQuest - Regression Trees, Clearly Explained (9 min, how trees handle continuous targets instead of categories): https://www.youtube.com/watch?v=g9c66TUylZ4
- Gradient Boosted Trees (StatQuest, 15 min, explains the sequential learning process clearly - watch after understanding decision trees and random forests): https://www.youtube.com/watch?v=3CC4N4z3GJc
- Gradient Boosting for classification (StatQuest, 11 min, extends the regression explanation to classification tasks): https://www.youtube.com/watch?v=jxuNLH5dXCs
- Gradient Boosted Trees https://www.youtube.com/watch?v=PxgVFp5a0E4

### Model interpretation - These are great and easy to understand

Tree-based algorithms are great because we can peak inside of them to see what they actually do, unlike neural networks.

- SHAP - Model Interpretation (visual explanation of SHAP values and how to read the plots we use in the notebook): https://www.youtube.com/watch?v=MQ6fFDwjuco
- Model Interpretable vs Explainable (short, clarifies the difference between models you can read directly vs models you need tools to explain): https://www.youtube.com/watch?v=VY7SCl_DFho

### Documentation

- scikit-learn Decision Trees (official docs, covers both classification and regression trees with API reference): https://scikit-learn.org/stable/modules/tree.html
- scikit-learn Random Forests (official docs, ensemble methods including random forests and gradient boosting): https://scikit-learn.org/stable/modules/ensemble.html
- XGBoost documentation (getting started guide and parameter tuning): https://xgboost.readthedocs.io/en/latest/
- LightGBM documentation (quick start and key differences from XGBoost): https://lightgbm.readthedocs.io/en/latest/
- SHAP documentation (Python library for model interpretation, includes tutorial notebooks): https://shap.readthedocs.io/en/latest/