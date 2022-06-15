# Sentiment Analysis on Amazon Reviews Dataset in PyTorch

## Project Overview
In this project, I’ll train LSTM networks on [Amazon Customer Reviews Dataset](https://www.kaggle.com/bittlingmayer/amazonreviews) to predict sentiment (Positive/Negative) of a review. You can run the codes on GPU to speed up the training process significantly. You can also use [My Notebook on Google Colab](https://colab.research.google.com/drive/1pFduf8iVOuFGm9-nkPxQEm3GniPhEBzw?usp=sharing) if your hardware is not powerful enough.

---

## Preparing the environment
**Note**: I have tested this project on __Linux__. It can surely be run on Windows and Mac with some little changes.

Before you can experiment with the code, you'll have to make sure that you have all the libraries and dependencies required to support this project. You will mainly need Python 3, PyTorch and its torchvision, OpenCV, Matplotlib, and tqdm.

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/iamirmasoud/amazon_sentiment.git
cd amazon_sentiment
```

2. Create (and activate) a new environment, named `sentiment_env` with Python 3.7. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	```shell
	conda create -n sentiment_env python=3.7
	source activate sentiment_env
	```
	
	At this point your command line should look something like: `(sentiment_env) <User>:amazon_sentiment <user>$`. The `(sentiment_env)` indicates that your environment has been activated, and you can proceed with further package installations.

6. Before you can experiment with the code, you'll have to make sure that you have all the libraries and dependencies required to support this project. You will mainly need Python3.7+, PyTorch and its torchvision, OpenCV, Matplotlib. You can install  dependencies using:
```
pip install -r requirements.txt
```

7. Navigate back to the repo. (Also, your source environment should still be activated at this point.)
```shell
cd amazon_sentiment
```

8. Open the directory of notebooks, using the below command. You'll see all the project files appear in your local environment; open the first notebook and follow the instructions.
```shell
jupyter notebook
```

9. Once you open any of the project notebooks, make sure you are in the correct `sentiment_env` environment by clicking `Kernel > Change Kernel > sentiment_env`.


### Data

Please download the [Amazon Customer Reviews Dataset](https://www.kaggle.com/bittlingmayer/amazonreviews) from Kaggle and put the `.bz2` files under the `data` subdirectory. The dataset contains a total of 4 million reviews with each review labeled to be of either positive or negative sentiment. However, I will only be using 100k reviews in this implementation to speed things up. Feel free to run it yourself with the entire dataset if you have the time and computing capacity. 


## Jupyter Notebooks
The project is structured as a series of Jupyter Notebooks that should be run in sequential order:

### [1_LSTM_Starter](1_LSTM_Starter.ipynb)

Before delving into the sentiment analysis task, I will review the basics of LSTM architecture in PyTorch in this notebook.

### [2_Amazon_Customer_Reviews_Sentiment_Analysis](2_Amazon_Customer_Reviews_Sentiment_Analysis.ipynb) 

In this notebook, I'll construct a Sentiment Analysis model that can be trained on the Amazon Review dataset.


## Results

I achieved an accuracy of **91.77%** on the test set with a simple LSTM architecture and only iterating 5 epochs! This shows the effectiveness of LSTM in handling such sequential tasks.

This result was achieved with just a few simple layers and without any hyperparameter tuning. There are so many other improvements that can be made to increase the model's effectiveness.

Some improvement suggestions are as follows:
- Running a hyperparameter search to optimize the configurations. 
- Increasing the model complexity
    - E.g. Adding more layers/using bidirectional LSTMs
- Using pre-trained word embeddings such as [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings

#### Beyond LSTMs
For many years, LSTMs has been state-of-the-art when it comes to NLP tasks. However, recent advancements in Attention-based models and Transformers have produced even better results. With the release of pre-trained transformer models such as Google’s BERT and OpenAI’s GPT, the use of LSTM has been declining. Nevertheless, understanding the concepts behind RNNs and LSTMs is definitely still useful, and who knows, maybe one day the LSTM will make its comeback?
