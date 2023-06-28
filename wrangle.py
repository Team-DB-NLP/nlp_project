import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
from pprint import pprint
import os
import time
import prepare
import json

import pandas as pd
import numpy as np

#see the data
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#play with words
import nltk
import re
from pprint import pprint

#split and model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def prepare_data(new_df):
    # Drop specific rows from the DataFrame
    new_df = new_df.drop([4, 26, 86, 143])
    # Reset the index of the DataFrame
    new_df = new_df.reset_index(drop=True)
    new_df.drop_duplicates(subset='readme_contents', inplace=True)
    new_df = new_df.dropna()
    new_df = new_df[new_df.language != 'Jupyter Notebook']
    new_df = new_df.reset_index(drop=True)
    new_df = new_df.drop('Unnamed: 0', axis=1)
    return new_df


def do_everything(codeup_df):
    cleaned_original = codeup_df['readme_contents'].apply(prepare.basic_clean)
    codeup_df['clean'] = cleaned_original
    tokenized_original = codeup_df['clean'].apply(prepare.tokenize)
    codeup_df['clean'] = tokenized_original

    codeup_df['clean'] = codeup_df['clean'].apply(prepare.remove_stopwords, extra_words = [], exclude_words = [])



    codeup_df['stemmed'] = codeup_df['clean'].apply(prepare.stem)
    codeup_df['lemmatized'] = codeup_df['clean'].apply(prepare.lemmatize)
    return codeup_df