# L03: Regression - Predicting Numbers with Gradient Descent

In Lesson 2, we trained an image classifier using fastai. We watched loss drop, accuracy rise, and a model go from random guesses to 90%+ accuracy on 37 pet breeds. But fastai did everything behind the scenes - the loss function, the training loop, the gradient updates. We saw the dashboard, not the engine.

This lesson opens the hood. We're going to build a model from scratch - pure Python and NumPy first, then PyTorch - and understand every single step of how it learns. The model itself is dead simple: linear regression, predicting a number from inputs. But the learning algorithm we use to train it - **gradient descent** - is the same algorithm that trains image classifiers, language models, and everything else in deep learning. Get comfortable with it here, and you've got the foundation for everything that follows.

## Regression vs Classification

In L2, the model picked a category (which breed?). That's **classification**. Now we're doing the other kind of supervised learning: **regression** - predicting a continuous number. House prices, temperatures, delivery times, salaries. The output isn't a label, it's a value on a scale.

The simplest regression model is a straight line: `prediction = weight x input + bias`. Two numbers - a slope and an intercept. The whole lesson is about answering: how do we find those two numbers automatically?

## The Loss Function

This is a concept we saw in L2 but never really looked inside. The **loss function** gives us a single number that measures how wrong our predictions are. Lower loss = better model.

For regression, the standard loss function is **Mean Squared Error (MSE)**: for each prediction, subtract the actual value, square it, then average all those squared errors. Squaring makes all errors positive and punishes big misses more than small ones. Being off by $200k on a house price is much worse than being off by $50k four times.

The entire goal of training is to minimize this loss. We want the weight and bias that make the MSE as small as possible. If you could somehow plot the MSE for every possible (weight, bias) combination, you'd get a bowl-shaped surface - the **loss landscape**. We need to find the bottom of that bowl. The notebook visualizes this loss landscape directly, and you'll even try to find the bottom by hand with sliders before we automate it.

## Derivatives and Gradients

To walk downhill, we need to know which direction downhill is. That's what the **derivative** tells us - the slope of the loss at our current position.

Think of standing on a hill blindfolded. You can feel the ground tilt under your feet. If it slopes left, move left to go down. If it's flat, you might be at the bottom. The derivative is that tilt - it tells you both the direction and the steepness.

When you have multiple parameters (multiple weights), the derivative of the loss with respect to each parameter is called a **gradient**. The gradient is just a collection of derivatives - one per parameter - telling you how to adjust each weight to reduce the loss.

This connects back to the **loss function**: the gradient depends on how wrong our predictions are. Bigger errors produce bigger gradients, which produce bigger weight updates. As the model improves and errors shrink, the gradients shrink too, and the updates become finer. The model naturally settles toward a minimum.

## Gradient Descent

This is the core algorithm. Everything in this course - from this simple line to the CNNs in lesson 11 - trains using gradient descent or a variant of it. The loop is:

1. Compute predictions with current weights (**forward pass**)
2. Compute the **loss** - how wrong are we?
3. Compute the **gradient** - which direction should each weight move?
4. Update weights: `weight = weight - learning_rate x gradient`
5. Repeat

The minus sign matters - the gradient points uphill (direction of steepest increase), so we go the opposite way. The **learning rate** controls the step size.

This loop repeats hundreds or thousands of times. Each pass, the loss gets a little smaller and the predictions get a little better. Eventually the weights converge to values that minimize the loss. The notebook builds this loop from scratch - first on a simple parabola where you can see each step, then on real housing data.

## The Learning Rate

The **learning rate** is the first **hyperparameter** you'll tune (remember hyperparameters from L2? Settings you choose before training, not learned from data). It controls how aggressively we update weights:

- Too small: the model inches toward the minimum, converging painfully slowly
- Too large: the model overshoots the minimum, bouncing wildly or diverging
- Just right: smooth, steady convergence

There's no formula for the perfect learning rate - you experiment. The notebook has interactive sliders where you can watch gradient descent with different learning rates. A good learning rate produces a smooth loss curve. A bad one makes the loss explode. This is one of those things that clicks way faster by seeing it than reading about it.

## Feature Scaling

If one input ranges from 0 to 15 and another from 0 to 10,000, the gradients for those features will be wildly different sizes. A learning rate that works for one is too big or too small for the other. The loss landscape gets stretched into a narrow valley, and gradient descent zigzags instead of heading straight to the minimum.

**Standardization** (subtract the mean, divide by the standard deviation) puts all features on the same scale. The loss landscape becomes a nice round bowl, and gradient descent converges much faster. Same algorithm, same starting point - dramatically different results. This is one of those "boring but critical" preprocessing steps that you'll do in every future lesson.

## The Training Loop

Worth repeating because it's that important:

```
for each iteration:
    predictions = model(inputs)               # Forward pass
    loss = loss_function(predictions, targets) # How wrong?
    gradients = compute_gradients()            # Which direction?
    parameters -= learning_rate * gradients    # Update
```

This pattern trains everything. Linear regression, neural networks, GPT. The model gets more complex. The **loss function** might change (MSE for regression, cross-entropy for classification - we'll see that in L4). The optimizer might get smarter (Adam instead of basic gradient descent - we'll see that in L5). But the loop structure never changes. We build it from scratch in NumPy, then show the PyTorch version doing the same thing with automatic gradient computation.

## Overfitting and Underfitting

In L2 we saw these as patterns in loss curves. Now we can explain why they happen and how to detect them.

**Underfitting**: the model is too simple for the data. A straight line through curved data will always be wrong - no amount of training fixes it. Both training loss and test loss stay high. The fix: a more complex model.

**Overfitting**: the model is too complex for the data. Given enough parameters, it can memorize every training point perfectly - and be useless on new data. Training loss drops to near zero, but test loss starts climbing. The model learned noise, not the real pattern. The fix: simpler model, more data, or regularization techniques (we'll cover those in later lessons).

The diagnostic is always the same: compare training loss to test loss. When the gap between them starts widening, you're overfitting. **Early stopping** - halting training when test loss starts rising - is the simplest defense.

## Bias and Variance

This is the theoretical framework underneath overfitting and underfitting:

- **Bias** is structural error. A straight line fitting curved data has high bias - it's systematically wrong regardless of how much data you have. The model's assumptions don't match reality.
- **Variance** is instability. A very complex model will look completely different depending on which data points it happens to train on. Train it on a slightly different sample and you get wildly different predictions. That's high variance - the model is fitting noise.
- **The tradeoff**: simple models have high bias but low variance (consistently wrong in the same way). Complex models have low bias but high variance (flexible but unstable). The sweet spot minimizes total error: bias + variance + irreducible noise.

In practice, you don't calculate bias and variance directly. You watch the training and test loss curves. High loss on both? Bias problem. Low training loss but high test loss? Variance problem. Same diagnostic as overfitting/underfitting, just with a deeper "why."

## From Linear to Neural

A straight line can only go so far. Linear regression combines features with weighted sums - it can never learn curves or interactions between features. The notebook ends by showing that a simple 2-layer neural network, using the exact same training loop and the exact same **loss function** (MSE), beats the linear model on the same data. Same **gradient descent**, same **learning rate** tuning - just a model that can bend.

That's the bridge to L4: we keep the loop, swap in a more powerful model, and switch from regression to classification.

## Terminology

| Term | What it means |
| --- | --- |
| **Regression** | Predicting a continuous number, not a category |
| **Loss function** | A single number measuring how wrong the model is. MSE for regression |
| **MSE (Mean Squared Error)** | Average of squared prediction errors - the loss function we use here |
| **Gradient / Derivative** | The slope of the loss at current parameters - tells you which direction reduces the loss |
| **Gradient descent** | The training algorithm: compute gradient, step opposite to it, repeat |
| **Learning rate** | Step size for weight updates - a hyperparameter you choose |
| **Forward pass** | Computing predictions from inputs using current weights |
| **Parameters / Weights** | Numbers the model learns from data during training |
| **Hyperparameter** | Settings you choose before training (learning rate, model size, epochs) |
| **Feature scaling / Standardization** | Normalizing inputs to similar ranges so gradient descent works well |
| **Overfitting** | Model memorizes training data instead of learning the real pattern |
| **Underfitting** | Model too simple to capture the real pattern |
| **Bias (statistical)** | Structural error from a model being too simple |
| **Variance (statistical)** | Model instability - predictions change a lot with different training data |
| **R-squared (R2)** | How much of the variance in the target the model explains (1.0 = perfect, 0 = useless) |
| **Epoch / Iteration** | One pass through the training loop. We use "iteration" here, "epoch" when using full datasets |

## https://www.youtube.com/watch?v=3dhcmeOTZ_Q&t=35sConnection to Previous and Next Lessons

**From L2:** We watched fastai's `fine_tune()` print epoch numbers, loss values, and accuracy. That output was a window into a training loop we couldn't see. Now we've built that loop ourselves - forward pass, loss, gradients, update. The loss dropping in L2's output? That was gradient descent running. The train/val split? Same golden rule we apply here.

**To L4:** Everything in this lesson was linear - weighted sums that can only draw straight lines. In L4, we go back to classification, but we look closer. Then, we’ll finally start talking about neural networks: sigmoid for classification, ReLU for non-linearity, hidden layers for learning complex patterns. The gradient descent loop doesn't change - we just plug in a more powerful model and swap MSE for cross-entropy loss.

## Resources

- Train / test split https://www.youtube.com/watch?v=SjOfbbfI2qY or https://www.youtube.com/watch?v=zAxuIlCBvOw→ Why do we split data?
- Linear regression short intro: https://www.youtube.com/watch?v=3dhcmeOTZ_Q&t=35s
- What is loss: https://www.youtube.com/watch?v=QBbC3Cjsnjg
- Gradient descent https://www.youtube.com/watch?v=sDv4f4s2SB8&t=85s (you dont have to think too much about the math, try to understand it more conceptually) - we’re checking how the loss is affected by moving the line (which actually means we’re updating something called weights / theta)
- Overfitting & underfitting: https://www.youtube.com/watch?v=B9rhzg6_LLw or this one https://www.youtube.com/watch?v=o3DztvnfAJg and this one https://www.youtube.com/watch?v=dBLZg-RqoLg
- StatQuest - Bias and Variance (7 min, connects overfitting/underfitting to the underlying theory): https://www.youtube.com/watch?v=EuBBz3bI-aA
- Derivatives - what is a derivative? (simple visual explanation of slope at a point): https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-1-new/ab-2-1/v/derivative-as-a-concept
- Feature scaling / normalization (why we need it for gradient descent, ~5 min, Andrew Ng draws the contour plots): https://www.youtube.com/watch?v=FDCfw-YqWTE

### Documentation and tutorials

- Google ML Crash Course - Linear Regression (interactive, good visual explanations of weights, bias, and loss): https://developers.google.com/machine-learning/crash-course/linear-regression