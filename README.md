# Repo_2
Week 2 — Classification & NLP Pipelines

This repository contains two production-grade Machine Learning and NLP pipelines covering Binary Classification and Natural Language Tokenization from scratch.

📌 Part 1: Logistic Regression Pipeline (notebook_1_logistic_regression.ipynb)Overview

An end-to-end Binary Classification pipeline implementing feature scaling, mathematical verification of the Sigmoid activation, Binary Cross-Entropy loss calculation, and 2D decision boundary visualization.

Pipeline Stages:

Data Preprocessing & Scaling:

Stratified split: Train (70%), Validation (15%), and Test (15%).

Standard scaling using StandardScaler fitted strictly on training data.

Model Training & Math Verification:

Training LogisticRegression with L2 regularization.

Manual computation of linear score $z = w^T x + b$ and Sigmoid probability $\sigma(z) = \frac{1}{1 + e^{-z}}$.

Cross-Entropy Loss computation:


$$\text{Loss} = -\frac{1}{N} \sum \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

Model Evaluation:

Metric assessment: Accuracy, Precision, Recall, F1-Score, and ROC-AUC at threshold $p = 0.5$.

2D Decision Boundary:

Hyperplane visualization separating Class 0 and Class 1 across feature spaces.

📌 Part 2: Tokenization & Subwords Pipeline (notebook_2_tokenization.ipynb)Overview

A comprehensive exploration of natural language tokenization pipelines, implementing basic tokenizers, Byte Pair Encoding (BPE) subword merging from scratch, context window chunking with overlap, and benchmark comparisons with tiktoken.

Pipeline Stages:

Text Preprocessing: Cleaning and normalizing literary text (alice.txt).

Custom Tokenizer (From Scratch): Regex-based token extraction, building vocabulary and bidirectional mappings (token_to_id and id_to_token).

Mini BPE (Byte Pair Encoding) Algorithm:

Splitting words into characters with boundary delimiters (</w>).

Iterative frequency counting and pair merging to generate a reproducible merge dictionary.

Context Window Chunking:

Splitting token streams into fixed context window sizes.

Preserving multi-token boundary Overlap to maintain semantic continuity.

Production Benchmark:

Side-by-side comparison table analyzing total token counts, unique vocabulary usage, and assigned IDs against OpenAI's tiktoken (cl100k_base).

🚀 Setup & Execution

Clone the repository:

git clone https://github.com/<your-username>/<your-repo-name>.git cd <your-repo-name> 

Install dependencies:

pip install numpy pandas matplotlib scikit-learn tiktoken 

Run notebooks:

