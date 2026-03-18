# L06: Full Tabular Pipeline

In Lesson 5, we opened the MLP's black box - tracing data through the forward pass, understanding backpropagation, dissecting the training loop, and comparing optimizers. We understand the mechanics of how a neural network learns. But we trained on a familiar dataset (Titanic) with clean data and simple features. Real-world data isn't like that. It has missing values, categorical strings, wildly different scales, and imbalanced classes. Before you can train anything useful, you need to deal with all of that.

This lesson is about the full pipeline - the complete workflow from raw messy data to a trained, evaluated model. Every step matters. Skip data exploration and you'll miss leakage. Skip normalization and your gradients will explode. Skip proper evaluation and you'll think your model is great when it's useless. This pipeline is what you'll reuse for every tabular project going forward, so we're building the habits right.

Think of it like building a web app. You wouldn't ship a FastAPI endpoint without input validation, error handling, and proper testing. The ML pipeline is the same discipline applied to a different domain: data prep is input validation, the DataLoader is pagination, the training loop is your processing pipeline, and evaluation is your test suite.

## Key Concepts

**Data Exploration** - before touching any model, you need to understand your data. Shape, distributions, correlations, missing values, target balance. This is where you catch problems early: is there data leakage? Is the target heavily imbalanced? Are there sentinel values pretending to be real data (like 999 meaning "not contacted")? Never skip this step.

**Missing Values** - real datasets have gaps. You have three options: drop rows (simple but wasteful), impute with mean/median/mode (preserves data but adds assumptions), or use a sentinel flag (useful when "missing" is itself informative). The right choice depends on how much data you have and why the values are missing. In the notebook, we'll encounter "unknown" values in multiple columns and make deliberate decisions about each.

**Normalization / Feature Scaling** - neural networks learn by multiplying inputs by weights and adjusting via gradients. If one feature ranges 0-5000 and another 0-7, the gradients will be dominated by the large feature. Standard scaling (subtract mean, divide by standard deviation) puts all features on the same playing field. Trees don't need this. Neural nets absolutely do.

**Categorical Encoding** - neural networks only understand numbers. Strings like "admin." or "married" need to be converted. Label encoding turns each category into an integer ID. For embedding-based MLPs (which we use in the notebook), these integer IDs become lookup keys into learnable embedding tables - small matrices where the network discovers its own representations for each category. One-hot encoding is the alternative, but it creates wide sparse inputs and defeats the purpose of embeddings.

**Embeddings** - each categorical column gets its own lookup table of learnable numbers. The job column with 12 categories might get a 12x6 embedding table. When the network sees "admin." (encoded as integer 0), it looks up row 0 and gets a 6-dimensional vector. These vectors start random and get optimized during training, just like weights. The network learns which jobs are similar to each other based on how they relate to the target.

**Train/Validation Split** - you split the data into two parts. The model trains on the training set and you evaluate on the validation set. If the model performs well on training data but poorly on validation data, it memorized instead of learned. The `stratify` parameter ensures both splits maintain the same class ratio - critical when your data is imbalanced.

**The Full Pipeline Pattern** - Preprocessing (clean, encode, scale) then Dataset/DataLoader (bridge from pandas to PyTorch tensors, batch the data) then Model (define the MLP architecture) then Training Loop (forward pass, loss, backward, update) then Evaluation (measure real performance). Each step is a building block. You'll repeat this pattern in every project.

**Class Imbalance** - what happens when 88% of your data is one class? A model that always predicts "no" gets 88% accuracy without learning anything. That's useless. We handle this with `pos_weight` in the loss function, which tells the model to pay more attention to the minority class. The model's raw accuracy might look lower, but it's actually learning meaningful patterns instead of taking the lazy path.

**Evaluation Beyond Accuracy** - accuracy alone is misleading with imbalanced data. We need the full picture:
- **Confusion matrix**: shows exactly where the model gets it right and wrong
- **Precision**: "of everything you predicted yes, how many were actually yes?"
- **Recall**: "of everything that was actually yes, how many did you catch?"
- **F1 score**: balances precision and recall into one number
- **AUC-ROC**: "if I pick a random positive and a random negative, how likely is the model to rank the positive higher?" Uses raw probabilities, not thresholds

**Cross-Validation** - a single train/val split can be lucky or unlucky. Cross-validation splits the data multiple ways and averages the results. More robust, especially with smaller datasets. Brief intro here, used more in later lessons.

**Hyperparameter Tuning** - the model's behavior depends heavily on choices you make before training: how many neurons, what learning rate, whether to use dropout, how much weight decay. These are hyperparameters - the model can't learn them, you choose them. The discipline is: change one thing, measure, compare. In the notebook we'll run multiple experiments and see that some settings overfit fast while others learn slowly but steadily.

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

### Course video

TBA

### Before the lesson

- StatQuest - Confusion Matrix (6 min, clear visual breakdown of true positives, false positives, and the full matrix): https://www.youtube.com/watch?v=Kdsp6soqA7o
- StatQuest - Sensitivity and Specificity (12 min, builds from the confusion matrix to precision/recall concepts): https://www.youtube.com/watch?v=vP06aMoz4v8
- StatQuest - Cross Validation (6 min, explains why a single split isn't enough and how k-fold works): https://www.youtube.com/watch?v=fSytzGwwBVw
- StatQuest - ROC and AUC (16 min, explains how to evaluate classifiers beyond accuracy using the ROC curve): https://www.youtube.com/watch?v=4jRBRDbJemM

### Documentation and tutorials

- PyTorch Dataset & DataLoaders tutorial (official docs, walks through building custom datasets and batching): https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- PyTorch classification from scratch (step-by-step dense neural network for classification): https://www.slingacademy.com/article/pytorch-classification-from-scratch-building-a-dense-neural-network/
- PyTorch for tabular data with categorical embeddings (the embedding approach we use in the notebook): https://yashuseth.wordpress.com/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/
- Google ML Crash Course - Classification metrics (interactive explanation of accuracy, precision, recall): https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall

### Quick videos

- DataLoaders explained (~10 min, visual walkthrough of PyTorch DataLoaders): https://www.youtube.com/watch?v=PXOzkkB5eH0
- NYC taxi fares with PyTorch (practical tabular example, end-to-end): https://francescopochetti.com/pytorch-for-tabular-data-predicting-nyc-taxi-fares/

### Additional reading

- Machine Learning Mastery - What is a Confusion Matrix (article, walks through the matrix with examples): https://machinelearningmastery.com/confusion-matrix-machine-learning/
