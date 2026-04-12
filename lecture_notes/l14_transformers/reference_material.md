# Phase 3 Reference Material: Transformers, Embeddings, and NLP

Material from Luke Ditria's PyTorch tutorials that's relevant to Phase 3 (LLMs, Embeddings, RAG).

## Sequential Models (section12)

Notebooks build autoregressive models from MLP through RNN to LSTM, then apply LSTMs to text classification, text generation, and question answering. Good progression for introducing sequence modeling before transformers.

Relevant notebooks:
- `Pytorch1_Autoregressive_MLP.ipynb` - autoregressive prediction with simple MLP
- `Pytorch2_Autoregressive_RNN.ipynb` - RNN for sequence prediction
- `Pytorch3_Autoregressive_LSTM.ipynb` - LSTM basics
- `Pytorch6_LSTM_Text_Classification.ipynb` - text classification with LSTM
- `Pytorch7_LSTM_Text_Generation.ipynb` - text generation with LSTM
- Uses SentencePiece tokenization and vocabulary building throughout

## Attention Mechanisms (section13)

Three notebooks building attention from first principles:

**Pytorch1_Attention_Basics.ipynb** - the standout notebook
- Builds attention from scratch: one-hot indexing -> soft lookup via softmax(Q*K^T)*V
- Implements multi-headed attention conceptually before using nn.MultiheadAttention
- Trains attention model to find closest matching MNIST images from a dataset
- Visualizations of what each query attends to
- Great pedagogical progression from hard indexing to soft attention to multi-head

**Pytorch2_LSTM_Attention_Text_Generation.ipynb**
- Combines LSTM with multi-head attention for text generation (AG News)
- Shows how attention "looks back" selectively instead of relying on LSTM memory
- Temperature-controlled sampling and attention weight inspection

**Pytorch3_CNN_Self_Attention.ipynb**
- Injects self-attention into CNN for global receptive field
- Visualizes learned attention maps showing which pixels attend to query locations
- Clear intuition for why attention helps (global context without deep stacking)

## Transformers (section14)

Full progression from classification to generation to vision:

**Pytorch1_Transformer_Text_Classification_Multi_Block.ipynb**
- Encoder-only transformer for AG News classification (4 categories)
- Sinusoidal positional embeddings, multi-block architecture, padding masking
- Pre-norm layer normalization, learnable output embedding vector
- Cosine annealing LR scheduler, token dropout regularization

**Pytorch2_Transformer_Text_Generation.ipynb**
- Decoder-only transformer with causal masking for next-token prediction
- Implements causal mask (triu) to prevent attending to future tokens
- Mixed precision training (torch.cuda.amp.autocast) with gradient scaling
- Temperature-controlled sampling with entropy tracking
- Good explanation of why causal masking is essential

**Pytorch4_Vision_Transformers.ipynb**
- ViT (Vision Transformer) - patches image into tokens, applies transformer
- Bridges computer vision and NLP approaches

## Embeddings and Semantic Similarity (section16)

**Semantic_Clustering.ipynb**
- Loads HuggingFace pretrained embeddings (BAAI/bge-small-en-v1.5)
- Extracts fixed-length embeddings from IMDB dataset, L2 normalizes
- K-means clustering on embeddings to discover semantic groupings
- t-SNE and PCA visualization of embedding space
- Identifies outliers within clusters by distance to centroid
- Production pattern: shows how pretrained embeddings capture semantic similarity

## Self-Supervised Pretraining (section06)

**Pytorch2_Unsupervised_PreTraining_Solutions.ipynb**
- Self-supervised learning via rotation prediction and puzzle-solving
- Pretrains ResNet18 on STL10 unlabeled data before downstream classification
- Shows how to leverage unlabeled data when you have limited labels
- Could be relevant if the course covers pretraining concepts

## How these map to the course plan

| Course Plan Lesson | Relevant Ditria Notebooks |
|---|---|
| LLM APIs + structured outputs | Not covered (his course is PyTorch-native, not API-focused) |
| Embeddings + semantic similarity | section16/Semantic_Clustering.ipynb |
| Brief RNN/attention history | section12 (RNNs/LSTMs), section13/Pytorch1_Attention_Basics.ipynb |
| Transformer fundamentals | section14/Pytorch1 (classification) + Pytorch2 (generation) |
| Vision Transformers | section14/Pytorch4_Vision_Transformers.ipynb |
| RAG | Not covered |

## Key teaching patterns to steal

1. **Attention from first principles** - the hard indexing -> soft attention -> multi-head progression in section13 is excellent
2. **Causal masking explanation** - section14 text generation notebook explains WHY clearly
3. **Embedding visualization** - t-SNE/PCA on HuggingFace embeddings with K-means is a clean demo
4. **Mixed precision training** - section14 shows practical AMP usage
5. **SentencePiece tokenization** - used consistently across NLP notebooks
