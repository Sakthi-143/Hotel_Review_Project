# Hotel_Review_Project
Business Objective:
	The dataset which consists of 20,000 reviews and ratings for different hotels and our goal is to examine how travelers are communicating their positive and negative experiences in online platforms for staying in a specific hotel and major objective is what are the attributes that travelers are considering while selecting a hotel. With this manager can understand which elements of their hotel influence more in forming a positive review or improves hotel brand image.
 

```markdown 
# Hotel Reviews Sentiment Analysis

This project aims to perform sentiment analysis on hotel reviews using various natural language processing techniques and machine learning algorithms. 

## Import Libraries

```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from afinn import Afinn
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from imblearn.over_sampling import SMOTE
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, plot_confusion_matrix
from pickle import dump, load
import warnings
warnings.filterwarnings("ignore")
```

## Descriptive Statistics

- Loaded the dataset from 'hotel_reviews.xlsx'
- Checked for null and duplicate values
- Described the dataset
- Visualized the distribution of ratings and sentiments

## Text Data Cleaning

- Applied text preprocessing techniques such as lemmatization, removing punctuation, converting text to lowercase, and removing stop words
- Calculated word count in cleaned reviews and visualized the distribution

## Affin Sentiment Analysis

- Used Afinn library to calculate sentiment scores
- Classified reviews into positive and negative sentiments based on the scores
- Visualized the distribution of sentiment scores

## Polarity Sentiment Analysis

- Used TextBlob library to calculate polarity scores
- Classified reviews into positive and negative sentiments based on the scores
- Visualized the distribution of polarity scores

## Vader Sentiment Analysis

- Used Vader SentimentIntensityAnalyzer to calculate compound sentiment scores
- Classified reviews into positive and negative sentiments based on the scores
- Visualized the distribution of compound sentiment scores

## Comparison between Different Sentiment Analysis Methods

- Compared the results of Affin, Polarity, and Vader sentiment analysis techniques
- Visualized the comparison using cross-tabulation

## Conclusion

In conclusion, this project demonstrates various approaches to sentiment analysis on hotel reviews and compares the effectiveness of different methods.
```

Feel free to adjust and customize the README as needed!
