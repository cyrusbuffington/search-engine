import os
import json
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from collections import defaultdict


def process_json_files(folder_path):
    'Yields data of each JSON file in folder_path'
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.json'):
                with open(os.path.join(root, filename), 'r') as f:
                    data = json.load(f)
                    yield data

def get_text_content(page):
    'Returns the text content of a JSON page object'
    soup = BeautifulSoup(page['content'], 'lxml')
    return soup.get_text()

def tokenize(text_content):
    'Tokenize the text content and return a list of tokens'
    tokens = []
   
    #Build token and process token when encounter non alphanum char
    token = ""
    for char in text_content:
        char = char.lower()
        if (ord(char)>=97 and ord(char)<=122) or (ord(char)>=48 and ord(char)<=57) or ord(char)==39:
            token += char
        elif len(token)>=3:
            tokens.append(token)
            token = ""
        else:
            token = ""
    if token:
        tokens.append(token)

    return tokens
    

def build_partial_indexes(folder_path, threshold):
    'Builds partial indexes and saves them to disk'
    inverted_index = defaultdict(lambda: defaultdict(int))
    file_counter = 0

    stemmer = PorterStemmer()

    total_pages = 0
    universal_tokens = set()

    for page in process_json_files(folder_path):
        total_pages += 1
        tokens = tokenize(get_text_content(page)) 

        for token in tokens:
            universal_tokens.add(token)
            stemmed_token = stemmer.stem(token)
            inverted_index[stemmed_token][page['url']] += 1

        if len(inverted_index) >= threshold:
            with open(f'indexes/partial_index_{file_counter}.json', 'w') as f:
                json.dump(inverted_index, f)
            inverted_index.clear()
            file_counter += 1

    if inverted_index:
        with open(f'indexes/partial_index_{file_counter}.json', 'w') as f:
            json.dump(inverted_index, f)
    
    #ANALYTICS
    print(f'Total pages: {total_pages}')
    print(f'Total words: {len(universal_tokens)}')


if __name__ == '__main__':
    build_partial_indexes('www_cs_uci_edu', 10000)
