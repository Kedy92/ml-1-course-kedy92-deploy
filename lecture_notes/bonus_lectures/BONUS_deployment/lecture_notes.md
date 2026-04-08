# Bonus: From Training to Production

This is a standalone lesson covering the model lifecycle after training. It answers the question students inevitably ask: "I trained a model, now what?"

The notebook is self-contained - it trains a small CNN on MNIST as a working example, then walks through saving, exporting, and serving it. No dependency on any specific lesson, though L5+ students will get the most out of it.

## What This Covers

**What IS a trained model?** A trained PyTorch model is four separate pieces that must stay in sync: the architecture (code), the weights (state_dict), the preprocessing pipeline (transforms/normalization), and the output mapping (class names). If any piece is wrong, you get silent wrong predictions - not errors. This is the most important concept in the notebook.

**state_dict deep dive.** What weights actually are - named tensors in an OrderedDict. The difference between learnable parameters (conv weights, linear weights) and tracked statistics (BatchNorm running mean/variance). Why `model.eval()` matters for the latter.

**Saving strategies.** `torch.save(state_dict)` vs `torch.save(model)` - why pickle is fragile, insecure, and version-dependent. The checkpoint pattern: bundling weights + metadata (preprocessing config, class names, epoch, accuracy) into one file. Resuming training from checkpoints.

**Export formats.** state_dict (.pth) for development. ONNX (.onnx) for production - framework-agnostic, faster inference via ONNX Runtime, runs without Python. TorchScript (.pt) for mobile/C++. The notebook exports to ONNX, runs inference with ONNX Runtime, and benchmarks the speedup.

**The inference pipeline.** The universal pattern: raw input -> preprocess -> tensor -> model -> post-process -> response. Common bugs demonstrated live: forgetting `model.eval()`, wrong normalization (silent wrong predictions), device mismatches.

**Serving with FastAPI.** Brief - students already know FastAPI/Docker from their devops background. Shows the ~25 lines of ML-specific code for a prediction endpoint. Key decisions: load at startup, CPU vs GPU, batching.

**MLOps basics.** Vocabulary and awareness only: model versioning, reproducibility, data drift, monitoring proxy signals, retraining loops. No implementation - just enough to know the terms and why they matter.

## Connection to Other Lessons

This lesson builds on concepts from across the course:

- **L5/L6**: The training loop produces weights. This lesson explains what those weights are and how to persist them.
- **L9/L11**: Image preprocessing (transforms, normalization) must match between training and inference. This lesson makes that explicit and shows what goes wrong when it doesn't.
- **L12**: Transfer learning involves loading pretrained weights - the save/load mechanics covered here are the foundation. L12 uses `torch.save(model.state_dict())` and `torch.load()` throughout.
- **L2**: fastai's `learn.export()` bundled everything into one pickle. This lesson explains what that hid and why pure PyTorch requires more manual work.

## When to Teach This

Flexible. Some options:
- After L6 (first full pipeline) - students have a trained model and want to know "now what?"
- After L8 (project assistance) - natural break before images
- After L12 (transfer learning) - connects to the save/load patterns used there
- As a standalone session whenever there's time

The notebook downloads MNIST and trains a model in the first few cells, so it doesn't depend on any prior notebook state.

## Notebook

`lessons/bonus_lectures/deployment/modified/deployment.ipynb`

## Resources

- PyTorch Saving and Loading Models (official): https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch ONNX Export (official): https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html
- ONNX Runtime documentation: https://onnxruntime.ai/docs/
- TorchServe (PyTorch model serving): https://pytorch.org/serve/
- MLflow (experiment tracking): https://mlflow.org/
