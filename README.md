# Practical Machine Learning for Programmers

Notebooks, projects, and datasets for learning practical machine learning with Python, PyTorch, and scikit-learn.

Website: [practical-ml.com](https://practical-ml.com)

## Repo structure

Each lesson folder follows this structure:

```
lesson_name/
  original/    <- Starter notebook (don't edit this one)
  modified/    <- Your working copy (edit, run, experiment here)
  example/     <- Sometimes exists: a completed example for reference
```

The `original/` folder is your reset point. If you break something in `modified/`, you can always copy the original back. Some homework folders include an `example/` folder with a completed version you can reference if you get stuck - try to solve it yourself first.

## Recommended setup: VS Code + local Python

The recommended way to work in this course is **locally in VS Code**. This gives you:
- Full file access (images, datasets, everything just works)
- Git integration for saving and syncing
- Agentic coding with Claude Code, Copilot, or similar tools
- No session timeouts, no upload friction

Most of the course runs fine on CPU. You don't need a GPU for L1-L7 (tabular data, small models). For the image-heavy lessons (L9-L12), training is slower on CPU but still works. If you want GPU acceleration for those lessons, see the Google Colab sections below.

### Setup with conda/mamba (recommended)

[Miniforge](https://github.com/conda-forge/miniforge) gives you `mamba` - a fast drop-in replacement for `conda`. Install Miniforge first, then:

```bash
git clone YOUR-REPO-URL
cd YOUR-REPO-NAME

# Create environment
mamba create -n ml-venv python=3.12 -y
mamba activate ml-venv

# Core ML stack (CPU-only, works for most of the course)
mamba install pytorch torchvision cpuonly -c pytorch -y
mamba install scikit-learn pandas numpy matplotlib seaborn jupyter -y

# Additional packages
pip install fastai xgboost lightgbm transformers datasets kaggle
```

> **Have a NVIDIA GPU?** Replace the pytorch line with:
> `mamba install pytorch torchvision pytorch-cuda=12.8 -c pytorch -c nvidia -y`
>
> **macOS Apple Silicon?** Replace the pytorch line with:
> `mamba install pytorch torchvision -c pytorch -y`

### Setup with pip

```bash
git clone YOUR-REPO-URL
cd YOUR-REPO-NAME
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn jupyter
pip install fastai xgboost lightgbm transformers datasets kaggle
```

### Running notebooks

Open the repo folder in VS Code, then open any `.ipynb` file. VS Code will ask you to select a kernel - pick your `ml-venv` environment. You might need to install the Jupyter extension for VS Code if you haven't already.

### Datasets

Small datasets are included in `data/`. Larger ones are downloaded automatically by the notebooks (e.g. fastai's `untar_data()`).

## When you need a GPU: Google Colab

For lessons that benefit from GPU acceleration (mainly L9-L12 with image training), you have two options. Note that both Colab options make it harder to use agentic coding tools like Claude Code, since those tools work best with a local environment where they can read files, run terminal commands, and interact with your full repo.

### Option A: VS Code + Colab extension

Use VS Code locally but connect to a Colab runtime for GPU-powered execution. You still edit files locally and use git normally, but the code runs on Google's servers.

1. Install the **Google Colab extension** in VS Code:
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Google Colab"
   - Install the one published by **Google** ([marketplace link](https://marketplace.visualstudio.com/items?itemName=Google.colab))
   - Video walkthrough: https://www.youtube.com/watch?v=EJvG8av1Z44

2. Open a notebook in VS Code and select **"Connect to Google Colab"** in the kernel picker

**Important: data files on Colab.** Your code runs on a remote server, not on your machine. Local files (datasets, CSV files) are NOT automatically available to the runtime. For notebooks that need local data files (L1, L3-L7), you need to upload the repo's `data/` folder to the Colab runtime:

- Right-click the `data` folder in VS Code's file explorer -> **"Upload to Colab Session"**
- This uploads it to `/content/data` on the runtime, which is where the notebooks expect it
- You need to do this once per session (sessions reset when you disconnect)

Notebooks that download their own data (L2, L9-L12) work without uploading anything.

Each notebook has a setup cell at the top that handles the path automatically - it uses `/content/data` on Colab and the relative path locally. Just run it first.

### Option B: Colab in the browser

If you don't want to install anything at all, you can open notebooks directly at [colab.google.com](https://colab.google.com). This is the most limited option - no agentic coding, no full repo structure, no Claude Code or Copilot.

**Setup:**
1. Go to [colab.google.com](https://colab.google.com) and sign in with your Google account
2. Go to [colab.google.com/github](https://colab.google.com/github) to connect your GitHub account
3. Check **"Include private repositories"** (important - the repo is private)
4. If prompted, grant access to the **UA-classroom** organization on GitHub

**Opening a notebook:**
1. In the GitHub tab, paste your repo URL
2. **Be patient** - the organization has many repos, searching can take 30-60 seconds
3. Browse to `modified/` and pick the notebook you want

**Saving:** File -> Save a copy in GitHub (commits back to your repo)

**Data files:** Same as the extension - notebooks that need local CSV files will require you to upload the data. You can upload files via the file browser panel on the left side of Colab, or use the setup cell instructions.

**Limitations:**
- Only opens one notebook at a time, not the full repo
- Some images from the repo won't display (code outputs and plots still work)
- No agentic coding tools (Claude Code, Copilot etc.) - this is a big limitation for this course
- Sessions timeout after ~90 minutes of inactivity
- Need to re-upload data files each session

**Troubleshooting:** If your repo doesn't show up, go to [github.com/settings/applications](https://github.com/settings/applications), find Google Colaboratory, and grant it access to the organization.

## Staying in sync with course updates

New lessons, fixes, and improvements are pushed regularly. You need to pull these updates into your repo.

### On GitHub (easiest)

If your repo on GitHub shows a banner saying **"This branch is X commits behind"**, click the **Sync fork** button and then **Update branch**. That's it.

If you don't see that button, you'll need to use the terminal method below.

### From the terminal

```bash
# One-time setup: add the book repo as upstream
git remote add upstream https://github.com/UA-classroom/practical-ml-book.git

# Pull latest changes (do this before each lesson)
git fetch upstream
git merge upstream/main
git push
```

Or just run the helper script:

```bash
./sync.sh
```

**Note:** If `git fetch upstream` gives you "repository not found", you might not have read access yet. Contact Utvecklarakademin to get access.

**Important:** If you only edit files in `modified/` folders (which you should), this will merge cleanly every time. If you do get merge conflicts, keep your version - your work matters more than the upstream update.

## License

"Practical Machine Learning for Programmers" by Tobias Fors

(c) 2025 Utvecklarakademin UA Aktiebolag. All rights reserved. See [LICENSE](LICENSE) for details.

This material may not be reproduced, distributed, or shared without written permission.
