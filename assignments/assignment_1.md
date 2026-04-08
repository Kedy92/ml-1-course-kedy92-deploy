# Assignment 1: Train, Deploy, Share

> **Note:** This is an example assignment from "Practical Machine Learning for Programmers."
> If you are completing this as part of a course, follow the submission and grading instructions
> from your official GitHub Classroom invitation.

**Covers:** Chapters 1-2

## What are we doing?

Train an image classifier (a neural network) on a dataset not used in the lesson notebooks, deploy it as a working API, and share it so others can try it out.

This is the same process from Chapter 2 - understand, prepare, train, evaluate, iterate, deploy - but on your own, with a different dataset. Just reading the lesson notebooks and running them does not make you learn. Learning kicks in when someone shows you a white canvas and says: "now draw." And then you realize - where do I start? That's when it clicks.

## The assignment

### 1. Train a model

Pick a dataset that wasn't used in the lesson notebooks (not Oxford Pets). The exercise notebook (`lessons/intro/homework/02_exercise_train_your_own.ipynb`) has several options ranging from easy to challenging. You can also find your own dataset.

Your notebook should show:

- **Data exploration** - what does the data look like? How many classes? Is it balanced?
- **Training** - train at least one model using transfer learning
- **Evaluation** - confusion matrix or most confused classes, top losses, sample predictions
- **At least one experiment** - change a hyperparameter (architecture, epochs, image size, etc.), compare the result to your baseline, and explain what happened

### 2. Deploy it

Get your trained model running as an API that accepts an image and returns a prediction. Use the `classifier_deploy/` starter from Chapter 2 as a base, or build your own.

It should:

- Accept an image upload
- Return the predicted class and confidence
- Run locally with `docker-compose up` (or equivalent)
- Optionally create a basic Streamlit frontend dashboard to let others try it out - prompt an AI to help you. Streamlit is very common among data scientists to prototype small projects.

You don't need to deploy to AWS or any cloud service.

### 3. Share it

Document your results:

- A screenshot or short recording of your app making a prediction
- What dataset you used and how many classes
- One thing that surprised you (a weird top-loss image, an unexpected confusion, a hyperparameter that helped or didn't)
- A link to the running API or Streamlit dashboard

## The notebook

Use the homework notebook from Chapter 2: `02_exercise_train_your_own.ipynb`. It has the dataset options, the process steps, and empty code cells for you to fill in.

Open the notebook, read through it, pick a dataset, and start prompting your way through each phase. The notebook walks you through the same six steps from Chapter 2 (understand, prepare, train, evaluate, iterate, ship) but this time the code cells are blank. Your job is to fill them in, step by step, using an agent to help you write the code.

Don't create a separate notebook from scratch. Work inside this one. The structure is already there - follow it, but remember: different datasets might have different challenges, and the agent will help you navigate that.

## How to work

Use Claude Code, Cursor, or any AI coding tool throughout. This is encouraged, not just allowed - it's how modern ML work is done.

But: **don't just accept what the agent gives you.** The goal is to understand what you're building. When the agent writes code, ask it to explain what each part does. When something doesn't work, try to understand why before asking for a fix. When it suggests an approach, ask yourself if it makes sense.

Good habits:

- **Explain things back to the agent.** "Let me try to explain what data augmentation does - correct me if I'm wrong..." This is one of the best ways to check your understanding.
- **Be critical of outputs.** If the agent generates a DataBlock, read it and make sure you understand each parameter. If something looks off, question it.
- **Ask it to research.** "What resize strategy should I use for satellite images and why?" is better than "write me a DataBlock."
- **Verify.** Always run `show_batch()` before training. Always look at your top losses after training. The agent can't see your outputs - you can.
- You can feed the agent the context of the book plan and the notebooks that will come. At this point you're creating a machine learning model using a neural network before you know how they work - that's fine! Focus on the process, and make sure the agent understands where you are in your learning journey.

## What you should produce

- **The completed homework notebook** (`02_exercise_train_your_own.ipynb`) with all code cells filled in, showing the full process: exploration, training, evaluation, experiment, results
- **A working deployment** (the `classifier_deploy/` folder with your model, or your own setup) with a README explaining how to run it
- **Your exported model** (`.pkl` file) - either in the repo or instructions for where to get it if it's too large for git

## Tips

- **Start with Imagenette** if you want the easiest path. The DataBlock setup is nearly identical to the lesson. But choose whatever sounds fun - the larger the dataset, the better GPU you'll need and the more time it will take to train.
- **Follow the notebook structure.** The homework notebook has phases (understand, prepare, train, evaluate, iterate, ship) with task descriptions and empty code cells. Work through them in order. Each phase builds on the previous one.
- **Start the deployment early.** The ML part is fun but the deployment is where people usually run out of time. Get `docker-compose up` working with the lesson's pet classifier first, then swap in your own model.
- **The experiment section matters.** Don't just train one model and call it done. Change something, compare, and write a sentence about what you learned. This is the habit that makes you an ML practitioner.
