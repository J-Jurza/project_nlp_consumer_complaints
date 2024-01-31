# Complaints Analysis using Sentence Transformer 📈

:warning: **Note**: This project is a **work in progress**. Expect ongoing updates and improvements as the project develops.

## Introduction 🌟

This project focuses on analysing consumer complaints through advanced NLP techniques, leveraging the power of Sentence Transformers for embedding generation, followed by clustering to categorise the complaints into distinct groups. The aim is to uncover patterns and insights that can help improve customer experience.

## Exploratory Data Analysis and Preprocessing 🕵️‍♂️

A simple NLP EDA is conducted on the consumer complaints dataset, with preprocessing steps to prepare the data for import into a Sentence Transformer. The preprocessing pipeline includes tokenization, stemming, and removal of stop words, setting the stage for effective embedding generation.

## Complaints Analysis NLP - Clustering Model 🧩

### Methodology 🔍

The methodology for clustering complaints data involves:
1. **Preprocessing**: Tokenizing, stemming, and removing stop words from the text data.
2. **Generating Embeddings**: Using a pre-trained Sentence Transformer model to produce sentence embeddings.
3. **Clustering**: Applying clustering algorithms such as KMeans, Hierarchical Clustering, DBSCAN, and Spectral Clustering. Tuning the number of clusters and hyperparameters through the elbow method, silhouette score, or dendrogram inspection.
4. **Analysis**: Identifying patterns, topics, and trends within the clusters to gain insights into consumer complaints.
5. **Evaluation**: Assessing the clusters based on their interpretability and coherence to inform business decisions.

### Types of Clustering 📊

The selection of a clustering algorithm is crucial and depends on factors like dataset size, number of clusters, similarity measures, and desired cluster interpretation. Commonly used algorithms for NLP tasks include KMeans, Hierarchical Clustering, DBSCAN, and Spectral Clustering.

## Dependencies

### General Packages

```python

### General packages
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pandas as pd
import numpy as np
import os
from datetime import datetime

### Modelling packages

from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

### Packages for saving models
import pickle
from joblib import dump

### Plotting packages
import matplotlib.pyplot as plt

```

