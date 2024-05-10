import os
import json
import hashlib
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
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
        if len(char) != 1:
            continue
        char = char.lower()
        if (ord(char)>=97 and ord(char)<=122) or (ord(char)>=48 and ord(char)<=57) or ord(char)==39:
            token += char
        else:
            tokens.append(token)
            token = ""
    if token:
        tokens.append(token)

    return tokens

def remove_fragment(url):
    'Removes the fragment part of a URL'
    return url.split('#')[0]
    

def build_partial_indexes(folder_path, threshold):
    'Builds partial indexes and saves them to disk'
    inverted_index = defaultdict(lambda: defaultdict(int))
    file_counter = 0

    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')  #Matches any sequence of word characters
    stemmer = PorterStemmer()

    total_pages = 0
    universal_tokens = set()
    url_dict = {} #Stores hashes of URLs to save disk space

    for page in process_json_files(folder_path):
        tokens = tokenizer.tokenize(get_text_content(page))

        url = remove_fragment(page['url'])
        url_hash = hashlib.md5(url.encode()).hexdigest() #Hash URL to save disk space
        url_dict[url_hash] = url #Store URL hash based on doc ID

        for token in tokens:
            token = token.lower()
            #universal_tokens.add(token)
            stemmed_token = stemmer.stem(token)
            universal_tokens.add(stemmed_token)
            inverted_index[stemmed_token][url_hash] += 1

        if len(inverted_index) >= threshold:
            with open(f'indexes/partial_index_{file_counter}.json', 'w') as f:
                json.dump(inverted_index, f) #Dump the inverted index
            with open(f'ids/url_dict_{file_counter}.json', 'w') as f:
                json.dump(url_dict, f)  #Dump the URL dictionary
            inverted_index.clear() #Clear the inverted index
            url_dict.clear()  #Clear the URL dictionary
            file_counter += 1

        total_pages += 1

    if inverted_index:
        with open(f'indexes/partial_index_{file_counter}.json', 'w') as f:
            json.dump(inverted_index, f)
        with open(f'ids/url_dict_{file_counter}.json', 'w') as f:
            json.dump(url_dict, f) 

    #ANALYTICS
    print(f'Total pages: {total_pages}')
    print(f'Total words: {len(universal_tokens)}')


if __name__ == '__main__':
    build_partial_indexes('developer', 200000)
