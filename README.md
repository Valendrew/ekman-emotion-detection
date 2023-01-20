# Sentiment Analysis on Ekman's Six Basic Emotions Corpus

This repository contains the work done for the Deep Learning exam in the Master's Degree in Artificial Intelligence at University of Bologna during the A.Y. 2021/2022.

However, it was one of the first approach with the Keras library, and in general with Deep Learning, for this reason some choices might be unpopular or not properly reported.

The work was done in a [Colab Notebook](https://colab.research.google.com/), hence it's recommended to run the notebook in the same environment. However, there shouldn't be any problems in other IPython Notebooks, except for some limitations (e.g. mounting the personal Google Drive storage).

The model achieves one of the highest score during the exam, 0.6070 for the macro F1-score.

## Prerequisites

The notebooks made some assumptions.

- Installed libraries are TensorFlow, Keras, Pandas, Matplotlib, Scikit-learn and Numpy.
- The tested Python version is 3.8.

Other libraries are installed directly in the notebook through pip. They are:

- Tensorflow addons: to compute an approximation of the F1-score as a metric of the network.
- Preprocessing libraries: Contractions, Emoji and Ekphrasis.

The model must respect guidelines made by the professor, such as a limited model size and achieve an higher macro F1-score.

## Implementation

### Dataset

The used dataset is "Ekman's size Basic Emotions", well-popular in sentiment classification tasks, which is structured in three columns, "Text", "Emotion and "Id", but the last one had been discarded. Each entry in the dataset contains a text and an associated emotion, among the six available (joy, neutral, surprise, anger, sadness, disgust and fear).

Some preprocessing techniques had been applied, such as transforming words in lowercase, replacing emojis with a text identifier (similar emojis are mapped to the same text) and some specific social network processing with the Ekphrasis library (e.g. elongated words). These were necessary to improve the model performance.

An insight on distribution of the classes, the number of samples, the train-test split are better described in the notebook.

### Model

The overall pipeline is structured as follows:
- TextVectorization layer for tokenizing words.
- Embedding layer loaded from the GloVE 6B vocabulary with 300 dimensions. It was set as non trainable afterwards, hence only the precomputed weights are considered.
- LSTM model.

The network architecture had Spatial Dropout applied on the Embedding layer, a few Bidirectional LSTM layers, dropout layers and a Dense layer for classifying.

Some regularizations and hyperparameters tuning were done to achieve higher scores and avoid overfitting.
