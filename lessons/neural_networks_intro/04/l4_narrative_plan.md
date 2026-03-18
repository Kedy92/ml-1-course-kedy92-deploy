# L4 Narrative Plan

## Part 1: What is Classification?

- Quick demos showing classification in different domains (medical, fraud, digits)
- Binary (two outcomes) vs multi-class (many outcomes) - we focus on binary here
- The common thread: labeled examples -> train -> predict on new data
- Key difference from regression: output is a probability, not a number

## Part 2: The Titanic Dataset

- Introduce the data, show the table, explain the features
- Visualize the classification problem: scatter plot with two colors mixed together
- Try drawing a manual boundary - it works but barely
- Explore survival patterns (gender, class, combinations)
- The model should discover these patterns on its own

## Part 3: Why Linear Regression Breaks

- Show binary data: dots only at 0 and 1, nothing in between
- Fit a straight line through it - it overshoots past 1 and below 0
- Show specific examples: "Fare=$350, output=1.2 - what does that mean?"
- We need outputs between 0 and 1 - we need probabilities

## Part 4: Sigmoid - Turning Numbers into Probabilities

- Introduce sigmoid: takes any number, squashes to 0-1
- Show the same examples through sigmoid: now they're valid probabilities
- Visualize: straight line vs S-curve on the same data
- This is an activation function - it activates the output of our linear model

## Part 5: Logistic Regression = Linear + Sigmoid

- The model: z = weighted sum of features, then sigmoid(z) = probability
- Same as L3: we start with RANDOM weights (show them!)
- Same gradient descent loop: forward pass, loss, gradients, update
- Show the weights at different stages of training - watch them converge
- Show predictions improving as weights get better

## Part 6: Cross-Entropy Loss

- Why not MSE? Show the math: MSE barely distinguishes "confidently wrong" from "somewhat wrong"
- Cross-entropy: confident wrong answers get massive penalty
- Visualize: cross-entropy curve vs MSE curve for wrong predictions
- This is the loss function for classification, just like MSE was for regression
- Different problems need different loss functions - this is a general principle

## Part 7: Evaluating and Interpreting

- Accuracy: how often is the model right?
- The learned weights tell a story: Sex has the biggest weight, matching "women and children first"
- The model discovered historical patterns from data alone - this is the satisfying moment
- Connect back to the survival patterns we explored earlier

## Part 8: The Decision Boundary

- Train a 2D model (Age, Fare) so we can visualize the boundary
- The boundary is a straight line - where sigmoid(z) = 0.5
- Everything on one side = predict survived, other side = predict died
- This is what classification looks like geometrically

## Part 9: The Limitation - Only Straight Lines

- Logistic regression can only draw straight boundaries
- Some problems need curves (e.g. inside/outside a circle)
- You could engineer polynomial features manually, but that doesn't scale
- What if the model could learn its own features?

## Part 10: Neural Networks (MLP) - The Teaser

- Add a hidden layer between input and output
- Hidden neurons learn feature combinations automatically
- Same training loop, same gradient descent, same loss function
- But now: random weights in TWO layers, gradient descent optimizes ALL of them
- Show parameter count: 5 vs 97
- Train and compare: MLP vs logistic regression

## Part 11: The Curved Boundary

- Visualize MLP's decision boundary side by side with logistic regression
- The MLP draws curves where logistic regression draws lines
- Why? Hidden layer transforms features into new space where a straight boundary = curve in original space

## Part 12: ReLU and Why It Matters

- Without ReLU: stacking linear layers = still linear
- With ReLU: introduces bends that allow curves
- Brief, don't go deep - L5 traces actual numbers through the layers

## Part 13: Summary and Bridge to L5

- Classification = probabilities, sigmoid, cross-entropy, decision boundaries
- Neural networks = hidden layers that learn features, can draw curves
- We've seen it WORK but treated it as a black box
- L5 opens the box: trace data through every layer, see what neurons compute

## Key Visualizations Needed

- 3-panel: binary data / linear regression overshooting / sigmoid fitting
- Cross-entropy vs MSE comparison plot
- Random weights -> trained weights (show the numbers changing)
- Decision boundary for logistic regression (straight line)
- Decision boundary comparison: logistic vs MLP (straight vs curved)
- ReLU function plot

## Key Tables Needed

- Raw linear output for specific fares (with "what does 1.2 mean?")
- Same outputs through sigmoid (now valid probabilities)
- Learned weights with interpretation (Sex = biggest negative weight)
- Model comparison: logistic vs MLP accuracy
