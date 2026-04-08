# L12: Transfer Learning and Object Detection

In L2, you fine-tuned a pretrained model with fastai: `learn.fine_tune(3)` and it worked. You had no idea why. In L9, you built an MLP on bird images from scratch - 25 million parameters, topped out at ~55%. In L10, you applied the same approach to CIFAR-10 and experimented with regularization. In L11, you built a CNN from scratch on the birds - 394K parameters, reached ~81%.

This lesson answers the question: what happens when we take a CNN that someone already trained on ~1.2 million images (ImageNet) and adapt it to our birds? The answer: dramatically better accuracy with minimal training, using a fraction of the compute. And now you'll understand exactly what's happening - because you built all the pieces in L9-L11.

This is the practical payoff of the entire image section. We also introduce object detection with YOLO - moving beyond "what's in this image?" to "where is everything?"

## Why Transfer Learning Works

A CNN trained on ImageNet has already learned a hierarchy of visual features:

- **Early layers** (layer 1-2): edges, gradients, color blobs. These are universal - useful for any image task.
- **Middle layers** (layer 3-5): textures, patterns, repeated structures. Still broadly useful.
- **Late layers** (layer 6+): parts, object components, high-level shapes. More task-specific.
- **Final layer** (classification head): maps features to ImageNet's 1000 classes. This is the only part that's useless for our task.

The key insight: we don't need to re-learn edges and textures. We just replace the final classification head with one that maps to our 5 bird classes, and fine-tune.

This is why the MLP from L9 was so wasteful - it had to learn everything from raw pixels, including basic edge detection, using only ~1000 training images. A pretrained CNN already knows what edges, textures, and shapes look like.

## Pretrained Models in torchvision

PyTorch's torchvision provides dozens of pretrained models. The ones we'll use:

**ResNet (Residual Networks):** The workhorse. ResNet-18 (11.7M params) through ResNet-152 (60M params). Skip connections allow training very deep networks without vanishing gradients. We use ResNet-18 in the notebook - fast to train and more than enough for our dataset.

**EfficientNet:** More recent, better accuracy-per-parameter than ResNet. EfficientNet-B0 is small and fast, B4-B7 scale up. Uses compound scaling (width, depth, resolution together). Note: the head is structured differently from ResNet (`model.classifier[1]` instead of `model.fc`).

Loading a pretrained model is one line:
```python
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
```

The `weights` parameter loads ImageNet-trained weights. Without it, you get random initialization (training from scratch).

## Feature Extraction vs Fine-Tuning

There are two strategies, and we'll try both:

### Feature extraction (freeze everything, train only the head)

Freeze all pretrained layers by setting `requires_grad = False`. Replace the final `fc` layer with a new one for our 5 classes. Train only the new layer.

This is fast (few trainable parameters) and works well when your dataset is small or very similar to ImageNet. The pretrained features are used as-is - like using a CNN as a fixed feature extractor.

### Full fine-tuning (unfreeze and adapt)

After training the head, unfreeze some or all pretrained layers and continue training with a lower learning rate. The pretrained weights get slightly adjusted to better fit your specific data.

The danger: if you use too high a learning rate, you destroy what the model already learned. That's why fine-tuning uses learning rates 10-100x smaller than training from scratch.

## Two-Phase Training

The standard recipe that works well in practice:

**Phase 1 - Train the head (2-5 epochs):**
- Freeze backbone
- Replace classification head
- Train with moderate learning rate (e.g. 1e-3)
- The head learns to map pretrained features to your classes

**Phase 2 - Fine-tune everything (5-15 epochs):**
- Unfreeze all layers
- Use a much lower learning rate (e.g. 1e-5 to 1e-4)
- Early stopping when validation loss plateaus

This is what fastai's `fine_tune()` does behind the scenes - now you know what it's doing.

## Discriminative Learning Rates

Not all layers should learn at the same speed. Early layers (edges, textures) need tiny adjustments. Late layers need bigger updates to adapt. The classification head needs the most freedom.

In PyTorch, you implement this with **parameter groups**:

```python
optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-3},
    {'params': model.fc.parameters(), 'lr': 1e-2},
])
```

Each group gets its own learning rate. Lower for early layers (preserve general features), higher for later layers (adapt to your task). This is what fastai calls "discriminative learning rates" - the concept Jeremy Howard explains in his course.

## Input Requirements

Pretrained models expect specific input formats. ImageNet models were trained with:

- Input size: 224x224 (some models accept other sizes)
- Normalization: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`

You must use the same normalization the model was trained with - not your dataset's stats. This is different from L9 where we computed per-channel stats from our bird images. The pretrained model's internal weights assume ImageNet normalization.

## The Bird Dataset Comparison

The narrative payoff of L9-L12:

| Model | Params | Val Accuracy | Lesson |
| --- | --- | --- | --- |
| MLP (from scratch) | ~25,000,000 | ~55% | L9 |
| CNN (from scratch) | ~395,000 | ~81% | L11 |
| ResNet-18 (fine-tuned) | ~11,700,000* | 90%+ | L12 |

*Most params are frozen during phase 1. Only ~2,500 params (new head) are trained initially.

The fine-tuned model should significantly beat both, despite training for just a few epochs on the same small dataset. This is the power of transfer learning - and the reason that in practice, you almost never train image models from scratch.

## Does Pretraining Source Matter?

The bird dataset is small and easy enough that any pretrained model crushes it. To see real differences between pretraining strategies, the notebook uses **Flowers102** (102 species, ~2,000 training images) - a harder fine-grained classification task.

Three identical ConvNeXt-Nano models are compared:
- **No pretraining** (random init): ~15% accuracy - can barely learn anything from 2,000 images
- **ImageNet-1k pretraining**: ~92% - the standard approach
- **ImageNet-12k pretraining** (broader source): ~95% - a few percent better from seeing more diverse images

The gap between no pretraining and any pretraining is enormous. The gap between different pretraining sources is smaller but real, especially on fine-grained tasks. The lesson: pretraining is non-negotiable for small datasets, and broader pretraining helps when classes are visually similar.

## Object Detection with YOLO

Everything so far has been **classification**: one image, one label. But real-world computer vision often needs more:

- **Detection**: where are the objects? (bounding boxes + class labels)
- **Segmentation**: which pixels belong to what? (pixel-level masks)

Detection is also a transfer learning story. A YOLO model pretrained on COCO (80 everyday object classes) can be fine-tuned to detect domain-specific objects - the same principle as taking ResNet from ImageNet to our birds.

### How YOLO works

**YOLO (You Only Look Once)** processes the entire image in a single forward pass, predicting bounding boxes and class labels simultaneously. Older approaches (R-CNN family) proposed regions first, then classified each one separately - much slower. YOLO's single-pass design makes it fast enough for real-time video.

Each detection has three components: bounding box coordinates (where), class label (what), and a confidence score (how sure the model is). Post-processing with **NMS (Non-Maximum Suppression)** removes duplicate detections of the same object based on how much the boxes overlap (**IoU** - Intersection over Union).

### The YOLO annotation format

Detection datasets need bounding box annotations - one `.txt` file per image:

```
class_id  x_center  y_center  width  height
0         0.45      0.62      0.30   0.55
```

All coordinates are normalized to 0-1 (relative to image size). A `dataset.yaml` file ties everything together: paths to images/labels, number of classes, and class names. The notebook shows both a real annotation file and the YAML.

### Fine-tuning YOLO

We use the `ultralytics` Python library, which wraps YOLO training, inference, and evaluation:

```python
from ultralytics import YOLO
model = YOLO('yolo11n.pt')  # pretrained on COCO (80 classes)
model.train(data='african-wildlife.yaml', epochs=10)
```

The notebook fine-tunes YOLO11-nano on the African Wildlife dataset (1,052 images, 4 classes: buffalo, elephant, rhino, zebra). COCO already knows some of these animals (elephant, zebra) but not others (buffalo, rhino) - fine-tuning adapts the model to this specific wildlife domain and adds the missing classes.

### YOLO versions and licensing

The YOLO family has evolved from YOLOv1 (2016) through many iterations. As of 2026, Ultralytics recommends YOLO11. Older tutorials reference YOLOv5 or YOLOv8 - the API is similar across versions.

Ultralytics YOLO is AGPL-3.0 for open-source use, with a separate commercial license. This matters if you deploy YOLO in a commercial product.

### Real-world use cases

Detection is one of the most commercially deployed ML tasks. The pattern is always the same: pretrained YOLO + domain-specific fine-tuning on a few hundred to a few thousand annotated images.

- **Manufacturing** - defect detection on production lines (scratches, missing components, cracks). BMW, Foxconn, and others use vision systems for automated inspection.
- **Workplace safety** - detecting whether workers wear required PPE (hard hats, vests, goggles). Alerts when someone enters a hazardous zone unprotected.
- **Retail** - monitoring store shelves for out-of-stock positions and misplaced products. Trax and Focal Systems deploy this for major retailers.
- **Healthcare** - counting blood cells in microscopy, detecting lesions in X-rays, locating tumors in MRI slices.
- **Agriculture** - John Deere's See & Spray uses detection to spray only weeds, reducing herbicide by 77%. Drones count fruit for yield estimation.
- **Traffic** - vehicle counting, pedestrian detection for autonomous driving (Tesla, Mobileye), smart city traffic management.
- **Document analysis** - detecting tables, figures, and text blocks in scanned documents for automated digitization.

[Roboflow Universe](https://universe.roboflow.com) has thousands of annotated datasets in YOLO format for these domains. The notebook covers several of these and points to public datasets students can try.

## Model Saving and Deployment

Model export, loading, serving (FastAPI), and MLOps basics are covered in a separate standalone notebook. See `lecture_notes/00_deployment/plan.md`. That material applies to any PyTorch model and can be taught whenever there's time in the schedule.

## Terminology

| Term | What it means |
| --- | --- |
| **Transfer learning** | Using a model trained on one task as the starting point for another |
| **Pretrained model** | A model with weights trained on a large dataset (e.g. ImageNet) |
| **Backbone / feature extractor** | All layers except the final classification head |
| **Classification head** | The final layer(s) that map features to class predictions |
| **Feature extraction** | Freezing the backbone, training only the head |
| **Fine-tuning** | Unfreezing some/all layers and training with a lower LR |
| **Discriminative LR** | Different learning rates for different layer groups |
| **ImageNet** | ~1.2M images, 1000 classes - the standard pretraining dataset |
| **YOLO** | Real-time object detection framework (ultralytics library) |
| **Bounding box** | Rectangle around a detected object: x, y, width, height |
| **Confidence score** | How sure the detector is that a box contains an object (0 to 1) |
| **IoU (Intersection over Union)** | How much two bounding boxes overlap - used to evaluate detection quality |
| **NMS (Non-Maximum Suppression)** | Post-processing step that removes duplicate detections of the same object |
| **COCO** | Common Objects in Context - 80-class dataset used to pretrain YOLO models |

## Notebooks

- **12_finetuning**: `lessons/neural_networks_images/12_finetuning_and_object_detection/modified/12_finetuning.ipynb` - fine-tuning pretrained models on the bird dataset, feature extraction vs full fine-tuning, discriminative LR, pretraining source comparison on Flowers102
- **object_detection**: `lessons/neural_networks_images/12_finetuning_and_object_detection/modified/object_detection.ipynb` - YOLO inference, annotation format, fine-tuning on African Wildlife + NEU steel defects, detection metrics (mAP, IoU)
- **finetuning_project**: `lessons/neural_networks_images/12_finetuning_and_object_detection/homework/original/finetuning_project.ipynb` - pick a dataset and compare from-scratch vs frozen backbone vs full fine-tuning (HOMEWORK)
- **object_detection_project**: `lessons/neural_networks_images/12_finetuning_and_object_detection/homework/original/object_detection_project.ipynb` - fine-tune YOLO on a new domain, evaluate and inspect results (HOMEWORK)

## Where This Fits

**From L11:** You built a CNN from scratch and got ~81% on the bird dataset. You understand convolutions, batchnorm, 1cycle, and why CNNs beat MLPs. Now you see what happens when you start with a CNN that already knows how to see.

**Back to L2:** In L2, you ran `learn.fine_tune(3)` with fastai and got a working pet classifier. Now you understand every piece of what that one line was doing: loading pretrained weights, freezing the backbone, replacing the head, training in two phases, using discriminative learning rates.

**The bigger picture:** L1-L12 is the complete image ML journey. You've gone from "what is ML?" to building, training, fine-tuning, and deploying image classifiers. The same transfer learning pattern applies beyond images - it's how GPT and BERT work for text (pretrain on massive data, fine-tune on your task). L13+ will explore that modern stack.

## Resources

### Videos

**Transfer learning concepts and practice:**
- Sebastian Raschka - Transfer Learning (7 min, concepts): https://www.youtube.com/watch?v=OkQRtm9JY1k
- Sebastian Raschka - Transfer Learning in PyTorch (11 min, code): https://www.youtube.com/watch?v=FaW9JCSJn2s
- Patrick Loeber - Transfer Learning PyTorch Beginner #15 (pure PyTorch, ResNet-18): https://www.python-engineer.com/courses/pytorchbeginner/15-transferlearning/
- Luke Ditria - Using transfer learning with neural networks: pytorch deep learning tutorial: https://www.youtube.com/watch?v=WhD4iDEW4w4
- Daniel Bourke - PyTorch Transfer Learning chapter (EfficientNet, detailed): https://www.learnpytorch.io/06_pytorch_transfer_learning/

**YOLO and object detection:**
- YOLO Object Detection - how it works (DataCamp, beginner-friendly overview of grid cells, bounding boxes, NMS): https://www.datacamp.com/blog/yolo-object-detection-explained
- YOLO - Intuitively and Exhaustively Explained (Towards Data Science, theory deep dive): https://towardsdatascience.com/yolo-intuitively-and-exhaustively-explained-83143925c7a9/
- Roboflow - How to Train YOLOv8 on a Custom Dataset (step-by-step practical tutorial): https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/
- LearnOpenCV - Train YOLOv8 on Custom Dataset (pothole detection, detailed walkthrough): https://learnopencv.com/train-yolov8-on-custom-dataset/
- Ultralytics - Training Custom Datasets in Google Colab: https://www.ultralytics.com/blog/training-custom-datasets-with-ultralytics-yolov8-in-google-colab

**Full courses with relevant sections:**
- Daniel Bourke - Learn PyTorch for Deep Learning (free 26h course, Chapter 6 covers transfer learning with EfficientNet): https://www.freecodecamp.org/news/learn-pytorch-for-deep-learning-in-day/
- fast.ai Practical Deep Learning - Lessons 3-6 (best conceptual explanations of discriminative LR and gradual unfreezing, code is fastai but concepts transfer): https://course.fast.ai/

### Documentation

**Transfer learning:**

- PyTorch Transfer Learning Tutorial (official, ResNet-18): https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- PyTorch Finetuning TorchVision Models (official, multiple architectures): https://docs.pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
- Layer-Wise Learning Rate in PyTorch (discriminative LR with parameter groups): https://www.kozodoi.me/blog/20220329/discriminative-lr
**YOLO and object detection:**
- Ultralytics docs (main reference, covers YOLO11/YOLO26): https://docs.ultralytics.com/
- Ultralytics training docs (all training parameters): https://docs.ultralytics.com/modes/train/
- YOLO dataset format (how to structure annotations): https://docs.ultralytics.com/datasets/detect/
- COCO8 - tiny 8-image detection dataset for testing (built into ultralytics): https://docs.ultralytics.com/datasets/detect/coco8/
- Roboflow Universe (free annotated datasets in YOLO format): https://universe.roboflow.com/
- V7 Labs - YOLO Algorithm Explained (good written explainer with diagrams): https://www.v7labs.com/blog/yolo-object-detection
