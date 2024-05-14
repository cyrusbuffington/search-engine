import os
import json
import pickle
import heapq
from posting import Posting
from porter2stemmer import Porter2Stemmer
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup
from collections import defaultdict
from operator import attrgetter

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


def remove_fragment(url):
    'Removes the fragment part of a URL'
    return url.split('#')[0]

def process_index_line(line):
    'Returns tuple of token and list of postings from a line in the index file'
    return line.split(':')[0], line.split(':')[1]#.split(',')

def dump_indexes(file_path, inverted_index):
    'Put inverted index on disk and merge with existing index'
    with open(file_path, 'w') as f:
        for token, postings in sorted(inverted_index.items()):
            f.write(f'{token}:')
            postings_list = [f'{posting.doc_id};{posting.term_freq}' for posting in sorted(postings, key=attrgetter('doc_id'))]
            f.write(','.join(postings_list))
            f.write('\n')

#TO DO, NEED TO MAINTAIN SORT ORDER
#BUILD LIST OF POSTINGS, SORT IT, THEN WRITE TO FILE
#DOES NOT NEED TO SORT EVERYTHING, JUST MERGE IN CORRECT ORDER

def merge_indexes(directory):
    'Merges all indexes in the directory into a single index file'
    #Get all index files in the directory
    index_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    #Open all files and add the first line of each to a heap
    heap = []
    files = [open(os.path.join(directory, f)) for f in index_files]
    for i, file in enumerate(files):
        line = file.readline().strip()
        if line:
            processed = process_index_line(line)
            heap.append((processed[0], processed[1], i))

    heapq.heapify(heap)
    current_token = '@'

    # Open the output file
    with open('merged_index.txt', 'w') as output_file:
        #While there are still lines in the heap
        while heap:
            #Pop the smallest line off the heap and write it to the output file
    
            next_token, postings, i = heapq.heappop(heap)

            if next_token == current_token:
                output_file.write(f',{postings}')
            else:
                output_file.write('\n')
                current_token = next_token
                output_file.write(f'{current_token}:')
                output_file.write(postings)

            #Add the next line from that file to the heap
            next_line = files[i].readline().strip()
            if next_line:
                processed = process_index_line(next_line)
                heapq.heappush(heap, (processed[0], processed[1], i))

    #Close all the input files
    for file in files:
        file.close()


def build_index(folder_path, threshold):
    'Builds partial indexes and saves them to disk'
    inverted_index = defaultdict(list)
    page_counter = 0

    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')  #Matches any sequence of alphanum characters
    stemmer = Porter2Stemmer()

    universal_tokens = set()
    doc_ids = []

    for page in process_json_files(folder_path):
        page_counter += 1
        tokens = tokenizer.tokenize(get_text_content(page))

        url = remove_fragment(page['url'])
        doc_ids.append(url)

        freqs = defaultdict(int)

        #Process each token in the page
        for token in tokens:
            token = token.lower()
            stemmed_token = stemmer.stem(token)
            universal_tokens.add(stemmed_token)
            freqs[stemmed_token] += 1

        for token, freq in freqs.items():
            inverted_index[token].append(Posting(page_counter - 1, freq)) #Add to inverted index

        #Dump indexes
        if page_counter % threshold == 0:
            dump_indexes(f'indexes/index_{page_counter // threshold}.txt', inverted_index)
            inverted_index.clear()

    with open(f'doc_ids.pkl', 'wb') as f:
        pickle.dump(doc_ids, f)  #Dump the doc_ids list to disk

    #Dump the remaining indexes
    if inverted_index:
        dump_indexes(f'indexes/index_{page_counter // threshold + 1}.txt', inverted_index)

    #Perform merging
    #merge_indexes('indexes')

    #ANALYTICS
    print(f'Total pages: {page_counter}')
    print(f'Total words: {len(universal_tokens)}')


if __name__ == '__main__':
    build_index('developer/DEV/mailman_ics_uci_edu', 10)
    #build_index('developer/DEV', 10000)
    merge_indexes('indexes')

