import requests
from requests import get
from bs4 import BeautifulSoup
import pandas as pd
from pprint import pprint
import os
import time
import random

def get_names_list():
    url = 'https://github.com/search?q=healthcare&type=repositories'
    headers = {"User-Agent": "how to fly a kite"}
    
    
    the_list_of_endings = []
    for j in range(2):
        
        time.sleep(5)
        response = get(url, headers = headers)
        soup = BeautifulSoup(response.content, 'html.parser')



        for i in soup.find_all('a', class_ = 'v-align-middle'):
            the_list_of_endings.append(i['href'])
        


        url = 'https://github.com' + soup.find_all('a', class_ = 'next_page')[0]['href']
    
    return the_list_of_endings