# AI Lab Experiments - Comprehensive Summaries
**Subject:** Artificial Intelligence (01CT0616)  
**Institution:** Marwadi University, Faculty of Technology, Department of ICT

---

## Experiment 01: Introduction to Machine Learning

**Experiment Number and Title:** Exp 01 - Introduction to Machine Learning

**Aim:** To learn the basics of Machine Learning and understand its fundamental concepts and applications.

**Key Theory Points:**
- Machine learning is a subset of AI that uses statistical techniques to enable computers to learn from data without explicit programming
- Machine learning tasks are classified into: Supervised Learning, Semi-supervised Learning, Reinforcement Learning, and Unsupervised Learning
- Major applications include Classification, Regression, Clustering, Density Estimation, and Dimensionality Reduction
- Key ML approaches: Decision Trees, Neural Networks, Support Vector Machines, Bayesian Networks, Genetic Algorithms, and Rule-Based Learning

**Methodology/Tasks:**
1. Load a dataset in the IDE
2. Observe and analyze statistics of all features (mean, std dev, min, max, quartiles)
3. Obtain the shape/dimensions of the dataset
4. Separate/extract all features from the dataset
5. Fill missing values using statistically relevant methods (median preferred over mean)
6. Visualize data using Box-Plots for each feature
7. Comment on IQR (Interquartile Range) and identify outliers
8. Analyze distribution and spread of data using IQR (Q3 - Q1)

**Key Concepts:**
- Supervised Learning: Trained with labeled data (input-output pairs)
- Unsupervised Learning: No labels; finds patterns in data
- Reinforcement Learning: Agent learns through interaction with environment via rewards/penalties
- IQR helps identify outliers: values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR

---

## Experiment 02: Linear Regression

**Experiment Number and Title:** Exp 02 - Linear Regression

**Aim:** To obtain the best fit line over single feature scattered datapoints using Linear Regression.

**Key Theory Points:**
- Linear regression assumes a linear relationship between input and output variables
- Errors are normally distributed, independent, and have constant variance (homoscedasticity)
- No strong multicollinearity should exist between features
- Mean Squared Error (MSE) is preferred over Mean Absolute Error (MAE) because it emphasizes larger errors and is differentiable everywhere
- Normal Equation: θ = (X^T X)^-1 X^T y provides a closed-form solution for optimal parameters

**Methodology/Tasks:**
1. Load libraries and packages
2. Load the dataset (e.g., advertising data with TV, Radio, Newspaper budgets vs. sales)
3. Analyze the dataset and its characteristics
4. Pre-process the data
5. Visualize the data with scatter plots (without best fit line)
6. Separate feature and prediction value columns
7. Write the Hypothesis Function: h(x) = θ₀ + θ₁x
8. Write the Cost Function (MSE)
9. Implement Gradient Descent optimization algorithm
10. Apply training to minimize loss
11. Find and visualize the best fit line
12. Observe cost function vs iterations learning curve

**Key Concepts:**
- Gradient Descent: Iteratively updates parameters to minimize cost
- Learning Curve: Shows reduction in error over iterations
- Best fit line represents optimized parameters with minimum error
- Outliers can be identified using IQR, boxplots, or Z-score methods

---

## Experiment 03: K-Nearest Neighbors (KNN)

**Experiment Number and Title:** Exp 03 - K-Nearest Neighbor Classification

**Aim:** To obtain the classification of various classes using k-Nearest Neighbor approach.

**Key Theory Points:**
- KNN is a non-parametric, lazy learner algorithm that stores entire training dataset
- Algorithm assumes similarity between new data and available cases
- K value is crucial: small K leads to overfitting; large K causes underfitting
- KNN can be used for both classification (majority voting) and regression (averaging)
- Common distance metrics: Euclidean, Manhattan, Minkowski, and Cosine Similarity

**Methodology/Tasks:**
1. Load libraries and packages
2. Load the dataset (e.g., Iris dataset)
3. Analyze the dataset
4. Pre-process and normalize the data
5. Visualize the data
6. Separate feature and prediction value columns
7. Select appropriate value of K (using cross-validation)
8. Calculate Euclidean distance of K nearest neighbors
9. Take K nearest neighbors based on calculated distances
10. Count data points in each category among K neighbors
11. Assign new data point to category with maximum neighbor count
12. Evaluate model performance

**Key Concepts:**
- Non-parametric: Makes no assumptions about underlying data distribution
- Lazy Learner: No learning during training; computation happens during prediction
- Euclidean Distance: √[(x₂-x₁)² + (y₂-y₁)²]
- For classification: Majority voting; For regression: Average of K neighbors
- Best K selected via cross-validation or grid search

**Advantages:** Simple, no training phase, works well with small datasets  
**Disadvantages:** Slow for large datasets, high memory requirements

---

## Experiment 04: Convolutional Neural Networks (CNN)

**Experiment Number and Title:** Exp 04 - Convolutional Neural Networks for Image Classification

**Aim:** To understand the process of convolution over images and apply it to classification problems.

**Key Theory Points:**
- CNN consists of three main layer types: Convolutional layers, Pooling layers, and Fully Connected layers
- Convolutional Layer: Applies filters (kernels) to extract features like edges, patterns, and textures
- Pooling Layer: Reduces spatial dimensions through Max Pooling (returns maximum value) or Average Pooling
- Max Pooling is preferred as it acts as noise suppressant and performs de-noising
- ReLU (Rectified Linear Unit) activation: Introduces non-linearity; removes negative values
- Fully Connected Layer: Takes feature maps and predicts the best label

**Methodology/Tasks:**
1. Load libraries and packages
2. Load the image dataset (e.g., Fashion MNIST)
3. Analyze the dataset
4. Normalize image data to 0-1 range
5. Pre-process the data
6. Visualize sample images
7. Design CNN model architecture with Conv, Pool, and FC layers
8. Write the cost function (categorical cross-entropy)
9. Implement optimization algorithm
10. Apply training to minimize loss
11. Apply dropout regularization to reduce overfitting
12. Observe cost function vs iterations learning curve
13. Compare training/validation accuracy with and without regularization

**Key Concepts:**
- Convolution: Filter slides over image to extract features
- Feature Map: Output showing locations and strength of detected features
- Dropout: Randomly turns off neurons to prevent overfitting
- Training without regularization: May overfit, validation accuracy plateaus
- Training with regularization: Better generalization, reduced gap between training and validation curves

**Dataset:** Fashion MNIST (28×28 grayscale images, 10 clothing classes)

---

## Experiment 07: N-gram Model for Language Modeling

**Experiment Number and Title:** Exp 07 - N-gram Model

**Aim:** To study N-gram models and understand language modeling and text prediction.

**Key Theory Points:**
- N-gram model predicts the most probable word following a sequence of N-1 words
- Based on probability distribution learned from training corpus
- Uses Markov Assumption to approximate probabilities: P(w_n|w₁...w_{n-1}) ≈ P(w_n|w_{n-1})
- Unigram: Uses only current word frequency
- Bigram: Considers previous word (most common)
- Trigram: Considers two previous words
- Applications: Speech recognition, machine translation, spelling correction, predictive text

**Methodology/Tasks:**
1. Prepare and preprocess text data
2. Tokenize text into words/characters
3. Remove stopwords and perform cleaning
4. Build N-gram frequency tables from corpus
5. Calculate conditional probabilities
6. Implement smoothing techniques (to handle unknown words)
7. Use Markov assumption to estimate probabilities
8. Generate next word predictions given seed text
9. Test with various seed phrases
10. Analyze probability distribution for word prediction

**Key Concepts:**
- Text Preprocessing: Tokenization, stopwords removal, normalization
- Probability Calculation: P('word₁ word₂ word₃') = P(word₁)×P(word₂|word₁)×P(word₃|word₂)
- For unknown words: Model may not predict well without smoothing techniques
- Example: "heavy rain" occurs more often than "heavy flood" in corpus → first is more probable

**Applications:** 
- Speech recognition (correcting noisy input)
- Machine translation
- Spell checking and error correction
- Sentiment analysis
- Language and dialect classification

---

## Experiment 08: Word Embedding for Text Vectorization

**Experiment Number and Title:** Exp 08 - Word Embedding

**Aim:** To study different word embedding techniques for representation of textual data in vectorized form.

**Key Theory Points:**
- Word embedding converts text data into numerical form for machine learning processing
- Bag of Words (BoW): Only considers word frequency, no word order
- TF (Term Frequency): Normalizes word frequency within a document
- TF-IDF (Term Frequency-Inverse Document Frequency): Gives importance to rare words while reducing weight of common words
- TF-IDF formula: TF-IDF(t,d) = TF(t,d) × log(Total Documents / Documents containing t)

**Methodology/Tasks:**
1. Prepare and preprocess text documents
2. Build vocabulary from corpus
3. Implement Bag of Words vectorization
4. Implement TF (Term Frequency) vectorization
5. Implement TF-IDF vectorization
6. Compare documents using cosine similarity
7. Analyze similarity scores between document pairs
8. Observe impact of different embedding methods on document similarity

**Key Concepts:**
- BoW: Simple frequency count; no semantic information
- TF: Normalizes BoW by document length
- TF-IDF: Better representation as it identifies important terms
- Cosine Similarity: Measures angle between vectors; ranges from -1 to 1

**Task Examples:**
- Take three documents and calculate similarity using all three methods
- Compare results and analysis from each approach

---

## Experiment 09: TextRank for Keyword Extraction

**Experiment Number and Title:** Exp 09 - TextRank for Keyword Extraction

**Aim:** Representation of a document with their keywords using TextRank algorithm.

**Key Theory Points:**
- TextRank is based on PageRank algorithm (originally for ranking webpages)
- PageRank: Calculates weight of webpages based on incoming links and their weights
- Formula: PR(e) = (1-d)/N + d × Σ(PR(i)/L(i)) where d is damping factor, L(i) is outgoing links
- TextRank adaptation: Treats words as nodes, co-occurrences as edges
- Identifies important words by analyzing their connections within text

**Key Components of TextRank:**
- Text preprocessing (tokenization, stopwords removal)
- Graph construction (words as nodes, co-occurrences as edges)
- PageRank algorithm application
- Ranking and extraction of top keywords

**Methodology/Tasks:**
1. Preprocess text data (tokenize, remove stopwords)
2. Build graph with words as nodes
3. Create edges based on word co-occurrences within context window
4. Apply PageRank algorithm iteratively
5. Initialize importance of each word to 1
6. Update importance scores based on connected words
7. Extract and rank top keywords
8. Analyze keyword importance scores

**Advantages of TextRank:**
- Unsupervised (no training data needed)
- Language-independent
- Simple and efficient
- Considers global word importance
- Robust and adaptable

**Limitations of TextRank:**
- No semantic understanding
- Sensitive to window size parameter
- Struggles with multi-word phrases
- Ignores part-of-speech (POS) tags

---

## Experiment 11: Content-Based Recommendation Systems

**Experiment Number and Title:** Exp 11 - Content-Based Recommendation Systems

**Aim:** To recommend items to users using content-based filtering approach.

**Key Theory Points:**
- Content-Based Filtering: Recommends items similar to those user previously liked
- Based on item features/properties, not other users' preferences
- Creates item profiles with important properties (for movies: genre, actors, director, year)
- Creates user profiles representing user preferences
- Calculates similarity between items using cosine similarity
- Uses TF-IDF vectorizer to convert text features to numerical form

**Methodology/Tasks:**
1. Load item dataset with features and ratings
2. Preprocess item content (genres, overview, keywords, etc.)
3. Combine relevant features for each item
4. Apply TF-IDF vectorization on text features
5. Create item feature vectors
6. Generate user profile from highly-rated items
7. Calculate cosine similarity between user profile and items
8. Rank items by similarity score
9. Recommend top N items not yet viewed by user
10. Evaluate recommendations

**Key Concepts:**
- Item Profile: Vector containing important properties of items
- User Profile: Aggregation of profiles of items user liked
- TF (Term Frequency): Number of times word appears in document
- IDF (Inverse Document Frequency): Measure of word significance across corpus
- Cosine Similarity: Measures angle between feature vectors

**Advantages:**
- No need for other users' data
- Recommends to users with unique tastes
- Can recommend new and popular items
- Provides explanations for recommendations

**Disadvantages:**
- Difficult to extract appropriate features
- Cannot recommend items outside user's interest area
- May cause over-specialization
- Requires detailed item data
- Ignores preferences of other users

---

## Experiment 12: Collaborative-Based Recommendation Systems

**Experiment Number and Title:** Exp 12 - Collaborative-Based Recommendation Systems

**Aim:** To recommend items to users using collaborative-based filtering approach.

**Key Theory Points:**
- Collaborative Filtering: Recommends items based on similar users' preferences
- Assumes users with similar interests have common preferences
- Does not require item feature information
- Uses user-item utility matrix showing user-item interactions
- Applies similarity measures (cosine similarity) between users
- Recommends items liked by similar users

**Methodology/Tasks:**
1. Load user ratings dataset (user-item matrix)
2. Preprocess rating data
3. Construct user-item utility matrix
4. Calculate similarity between target user and all other users using cosine similarity
5. Find K most similar users
6. Calculate weighted ratings for unrated items based on similar users
7. Rank items by weighted ratings
8. Recommend top N items not yet viewed by user
9. Evaluate recommendation quality

**Key Concepts:**
- User-Item Matrix: Rows = users, Columns = items, Values = ratings
- Cosine Similarity: Measures similarity between two user preference vectors
- Weighted Ratings: Average ratings of similar users, weighted by their similarity
- Cold Start Problem: New users/items with insufficient data for recommendations

**Advantages:**
- No need for item feature information
- Can recommend diverse and unexpected items
- Improves with more user data
- Provides personalized recommendations
- Captures user behavior effectively

**Disadvantages:**
- Cold start problem (new users or items)
- Requires large amount of user data
- Data sparsity issue (few ratings available)
- Scalability issues with large datasets
- May be inaccurate with insufficient data

---

## Experiment 13: Reinforcement Learning for Shortest Path

**Experiment Number and Title:** Exp 13 - Reinforcement Learning

**Aim:** To train an agent to find the shortest path in an environment using Reinforcement Learning.

**Key Theory Points:**
- Reinforcement Learning: Agent learns by interacting with environment and receiving rewards/penalties
- Learning is based on trial-and-error; no predefined training dataset
- Goal: Maximize cumulative reward over time
- Different from Supervised Learning: No correct input-output pairs provided
- Agent learns optimal policy mapping states to actions

**Reinforcement Learning Elements:**
- Policy: Defines agent's behavior (mapping states to actions)
- Reward Function: Provides numerical score based on environment state and action
- Value Function: Measures long-term reward from a state
- Model of Environment: Used for planning

**Types of Reinforcement:**
- Positive Reinforcement: Event increases behavior frequency and strength
- Negative Reinforcement: Removing negative condition strengthens behavior

**Main Components:**
- Agent: Learner/decision-maker
- Environment: System agent interacts with
- State: Current situation of agent
- Action: Choices available to agent
- Reward: Feedback after taking action

**Methodology/Tasks:**
1. Define environment (grid/graph with start, goal, obstacles)
2. Initialize Q-matrix (state-action value table)
3. Set discount factor (gamma) for future rewards
4. Implement Q-Learning algorithm
5. For each episode:
   - Start at initial state
   - Select action (exploration vs exploitation balance)
   - Observe next state and reward
   - Update Q-value: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
   - Continue until goal state reached
6. Repeat episodes until Q-matrix converges
7. Extract optimal path using maximum Q-values

**Key Concepts:**
- Q-Learning: Learns Q-values (action quality) in different states
- Exploration: Trying new actions to discover better solutions
- Exploitation: Using known high-reward actions
- Balance required between exploration and exploitation
- Discount Factor (γ): Weight given to future rewards (0-1)
- Learning Rate (α): Controls how much new information overrides old

**Advantages:**
- Solves complex problems conventional techniques cannot
- Handles non-deterministic environments
- Flexible; can combine with other ML techniques
- Model can correct training errors

**Disadvantages:**
- Not suitable for simple problems
- Needs lots of data and computation
- Highly dependent on reward function design
- Difficult to debug and interpret

**Example:** Robot finding shortest path to diamond while avoiding fire obstacles

---

## Summary Table

| Exp | Title | Key Technique | Main Application |
|-----|-------|---------------|------------------|
| 01 | Introduction to ML | Data Analysis | Understanding ML basics |
| 02 | Linear Regression | Gradient Descent | Continuous value prediction |
| 03 | K-Nearest Neighbors | Distance Metrics | Classification |
| 04 | CNN | Convolution & Pooling | Image Classification |
| 07 | N-gram Model | Probability Distribution | Language Modeling |
| 08 | Word Embedding | TF-IDF | Text Vectorization |
| 09 | TextRank | PageRank Algorithm | Keyword Extraction |
| 11 | Content-Based RS | Cosine Similarity | Item Recommendation |
| 12 | Collaborative RS | User Similarity | Personalized Recommendation |
| 13 | Reinforcement Learning | Q-Learning | Path Optimization |

---

**Document Generated:** April 21, 2026  
**Institution:** Marwadi University, Department of ICT  
**Subject:** Artificial Intelligence (01CT0616)
