import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
from pprint import pprint
import os
import time
import random


def headers_function():
    the_list = ['pandas', 'apples', 'divante222', 'cobra', 'circle']
    
    return random.choice(the_list)


def get_names_list():
    url = 'https://github.com/search?q=healthcare&type=repositories&p=15'
    
    
    
    the_list_of_endings = []
    for j in range(5):
        headers = {"User-Agent": headers_function()}
        time.sleep(5)
        response = get(url, headers = headers)
        soup = BeautifulSoup(response.content, 'html.parser')



        for i in soup.find_all('a', class_ = 'v-align-middle'):
            the_list_of_endings.append((i['href'][1:]).strip())
        


        url = 'https://github.com' + soup.find_all('a', class_ = 'next_page')[0]['href']
    
    return the_list_of_endings