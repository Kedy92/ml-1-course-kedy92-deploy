# Bonus: Collaborative Filtering & Recommendation Engines

Standalone topic that bridges tabular ML (L6) and the modern stack (L14+). Embeddings appear here as learned representations before the NLP/LLM context.

## Primary Resource

Jeremy Howard — Practical Deep Learning, Lesson 7 (start from 1:02:00):
https://www.youtube.com/watch?v=p4ZZq0736Po&list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU&index=7

Arguably the best lesson in the series — Jeremy explains collab filtering using PyTorch, stepping through the math in a spreadsheet before coding. The way he builds from Excel SGD to embeddings to a full neural net version is exceptionally well done.

## Topics

**The recommendation problem** — sparse user-item matrix, predict missing ratings. "If Alice rated movies A, B, C and Bob rated A, B, D — what would Alice rate D?"

**Latent factors** — randomly initialized embedding vectors for users AND items. Both are learnable parameters, not inputs. The model discovers hidden taste dimensions through training.

**Dot product as similarity** — no hidden layers, no ReLU, purely linear. It works because this is matrix factorization, not function approximation. The dot product measures how well a user's taste vector aligns with a movie's feature vector.

**Biases** — some users rate generously, some movies are universally liked. Biases capture these baseline tendencies independent of the match quality.

**Weight decay / regularization** — prevent overfitting on sparse data. Without it, the model memorizes known ratings instead of learning generalizable patterns.

**Interpreting embeddings** — PCA to visualize learned movie clusters. Movies that end up close in embedding space share something the model discovered (genre, mood, era — without ever being told these categories exist).

**CollabNN** — upgrade from dot product to a real neural network with hidden layers. Allows nonlinear interactions between user and item embeddings. More powerful but needs more data.

**Cold start problem** — new users/items have random embeddings and no rating history. Need content features (genre, description) to bootstrap until enough interactions exist.

## Key Concept: Latent Factors vs Content Features

**Latent factors only:** just embeddings + dot product. Discovers hidden patterns from ratings alone. Can't handle new users/items.

**Content features only:** traditional ML on known attributes (genre, age, etc.). Handles cold start but misses subtle taste patterns.

**Hybrid** (what Netflix/Spotify/YouTube do): embeddings for IDs + real features + neural network. Best of both worlds.

## Notebook

`lessons/recommendation_engines/collab_filtering_fastai.ipynb`
