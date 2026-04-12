# L13: Collaborative Filtering & Recommendation Engines

Open Netflix and the homepage is different for every user. Spotify's Discover Weekly finds music you didn't know you'd love. Amazon shows "customers who bought this also bought..." before you even finish scrolling. All of these are recommendation systems, and most of them are built on the same core idea: collaborative filtering.

The premise is surprisingly simple. If thousands of users have rated movies, and Alice's ratings look a lot like Bob's, then movies Bob loved that Alice hasn't seen are probably good recommendations for Alice. You don't need to know anything about the movies themselves - no genres, no actors, no descriptions. Just the pattern of who liked what.

This turns out to be one of the most commercially important problems in ML. Netflix estimated that their recommendation system is worth over $1 billion per year in retained subscriptions. Spotify, YouTube, Amazon, and TikTok are built around the same fundamental challenge: given a history of user interactions, predict what each user would engage with next.

## What Makes This Different From Previous Lessons

In every lesson so far, our models received meaningful inputs: pixel values, passenger ages, housing prices, image features. The model learned a function from those inputs to an output.

In collaborative filtering, the main inputs are just IDs. "User 42 rated Movie 100 a score of 3." The number 42 means nothing to the model. The number 100 means nothing. The model has to discover, entirely from the pattern of ratings, what those IDs represent. It does this by assigning each user and each movie a vector of learned numbers - an embedding - and adjusting those numbers until the math works out.

This is the first time we build representations from scratch rather than working with data that already has structure. That idea - learning representations from raw interactions - is one of the most powerful concepts in modern ML. Word embeddings (word2vec), graph embeddings, and the token embeddings inside language models all work the same way.

## The Core Idea

Imagine you could describe every movie with a few hidden numbers - how cerebral it is, how action-packed, how dark. And imagine every user had a matching set of numbers describing their taste. To predict whether a user would like a movie, just multiply the matching numbers together and add them up (a dot product). If the numbers align, the score is high.

The problem: we don't have those numbers. Nobody has scored every movie on "cerebral" or interviewed every user about their preferences. And we wouldn't even know which qualities to measure.

The solution: start with random numbers, use gradient descent to nudge them toward values that make the dot products match real ratings. The same SGD loop from L3 and L5. After training, the numbers mean something - but nobody defined what. The model discovered whatever hidden dimensions best explain the rating patterns. These are called latent factors.

## What the Notebook Covers

The notebook builds a full recommendation system from scratch in PyTorch, progressing through increasingly capable models:

**The data**: MovieLens 100K - 100,000 ratings from 943 users on 1,682 movies. We visualize the sparse user-movie matrix and set up the "fill in the blanks" framing.

**Embeddings**: What they are (learned lookup tables), how they work (nn.Embedding), and the terminology (embedding matrix, embedding, latent factor). We also include an Excel spreadsheet where you can see the calculations live and change values to watch predictions update.

**DotProduct model**: Two embedding matrices and a dot product. The simplest possible collaborative filtering model. We trace through the math, train it, and watch it overfit.

**Training dynamics**: Why learning rate matters (1e-2 overfits instantly, 1e-3 actually trains), how many latent factors to use, and what happens when you add biases for user generosity and movie quality.

**Regularization**: Weight decay, early stopping, dropout on embeddings. Each technique is tested experimentally and we compare results side by side.

**Using the model**: Top picks for a user, "because you watched X" using cosine similarity, filling in the original sparse matrix, interpreting movie biases, and visualizing the embedding space with PCA.

**Beyond dot products**: CollabNN replaces the dot product with a neural network (concatenate embeddings, feed through an MLP). It's more flexible but doesn't automatically win on small datasets.

**Hybrid models**: Adding real metadata (age, occupation, genres, release year) alongside the learned embeddings. This is how production systems actually work - they combine the collaborative signal with known features. We show that the hybrid approach helps most when interaction history is sparse.

**Cold start**: What happens when a new user or movie has no history. Strategies and limitations.

## Key Concepts

**Collaborative filtering** predicts preferences from collective interaction patterns. The "collaborative" part means many users' behavior teaches the model what each item is like.

**Embeddings** are learned vector representations. Each user and each movie gets a row of numbers that started random and was shaped by training. Similar items end up with similar vectors.

**The dot product** measures alignment between a user's taste vector and a movie's quality vector. High alignment = high predicted rating.

**Cosine similarity** measures whether two vectors have the same pattern of highs and lows, regardless of scale. This powers "because you watched X" recommendations.

**Matrix completion** is the formal name for what collaborative filtering does: given a sparse matrix with many missing entries, estimate the missing values.

**Hybrid models** combine learned embeddings with known metadata, feeding everything through a neural network. This is the standard architecture in production recommendation systems.

## Where This Fits

**From L12:** L12 closed the image/CNN arc with transfer learning and object detection. The training loop, loss curves, and regularization techniques all carry forward. The new idea here is learned representations for discrete entities (IDs) rather than continuous inputs (pixels).

**To L14+ (Embeddings & NLP):** The embeddings we learn here for users and movies are the same concept that powers word embeddings (word2vec, GloVe) and modern language models. In L14, we'll see embeddings applied to text - same idea, different domain. The collaborative filtering intuition ("similar items end up with similar vectors") transfers directly.

**The bigger picture:** Collaborative filtering is matrix factorization. The dot product model decomposes a sparse interaction matrix into two low-rank embedding matrices. This same decomposition appears across ML: word2vec, PCA, topic models, and graph embeddings. Understanding it here prepares you for all of them.

## Notebooks

- **Lesson**: `lessons/recommendation_engines/lesson/modified/collab_filtering_pytorch.ipynb` - full guided walkthrough
- **Excel demo**: `lessons/recommendation_engines/lesson/modified/collab_filter_demo.xlsx` - interactive spreadsheet, will use during lecture

## Resources

### Primary Resource

Jeremy Howard - Practical Deep Learning, Lesson 7 (start from 1:02:00) and part 8 up until CNNs:
https://www.youtube.com/watch?v=p4ZZq0736Po&list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU&index=7

Arguably the best lesson in the series - Jeremy explains collab filtering using PyTorch, stepping through the math in a spreadsheet before coding. Our notebook follows a similar arc but diverges in some areas - we focus more on experimentation and hybrid models.

### Deep Dives: From This Notebook to Real Systems

Our notebook teaches the scoring core: embeddings, latent factors, hybrid models, cold start intuition. Production systems add retrieval, ranking, re-ranking, implicit feedback, business rules, and much more. These resources show what gets built on top, which is another me for me to tell you that the notebook sets you on a great foundation, but in large companies who run recommendation engines there are usually more steps to it, which is one of the reasons recommendation engines is a niche in itself, people specialize in it and dive even deeper:

**Netflix / industry overview**
Moumita Bhattacharya - Recommender and Search Ranking Systems in Large Scale Real World Applications
https://www.infoq.com/presentations/recommender-search-ranking/

**Spotify / production ML platform**
Josh Baer - Powering Spotify's Audio Personalization Platform
https://www.thestrangeloop.com/2022/powering-spotifys-audio-personalization-platform.html

**YouTube: candidate generation + ranking**
Paul Covington, Jay Adams, Emre Sargin - Deep Neural Networks for YouTube Recommendations
https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/

**Google: hybrid sparse + dense features**
Heng-Tze Cheng et al. - Wide & Deep Learning for Recommender Systems
https://research.google/pubs/wide-deep-learning-for-recommender-systems/

**Netflix: what a modern stack grows into**
Netflix TechBlog - Foundation Model for Personalized Recommendation
https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39

### Documentation

- PyTorch nn.Embedding docs: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- MovieLens dataset: https://grouplens.org/datasets/movielens/