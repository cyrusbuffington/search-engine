import os
import json
import pickle
import heapq
import warnings
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
    print(page['content'])
    print(soup.get_text())
    return soup.get_text()


def remove_fragment(url):
    'Removes the fragment part of a URL'
    return url.split('#')[0]


def process_index_line(line):
    'Returns tuple of token and list of postings from a line in the index file'
    return line.split(':')[0], line.split(':')[1]


def dump_indexes(file_path, inverted_index):
    'Put inverted index on disk and merge with existing index'
    with open(file_path, 'w') as f:
        for token, postings in sorted(inverted_index.items()):
            f.write(f'{token}:')
            postings_list = [f'{posting.doc_id};{posting.term_freq}' for posting in sorted(postings, key=attrgetter('doc_id'))]
            f.write(','.join(postings_list))
            f.write('\n')


def load_pickle_file(file_path):
    'Loads the pickle file back to memoryfrom disk'
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def get_postings(file_path, token, token_positions):
    'Returns the postings list for a token from the index file'
    if token not in token_positions:
        return []
    with open(file_path, 'r') as f:
        f.seek(token_positions[token])
        line = f.readline().strip()
        postings = process_index_line(line)[1].split(',')
        postings = [Posting(int(posting.split(';')[0]), int(posting.split(';')[1]), doc_freq=len(postings)) for posting in postings]
        return postings


def merge_indexes(directory):
    'Merges all indexes in the directory into a single index file'
    #Get all index files in the directory
    index_files = ([f for f in os.listdir(directory) if f.endswith('.txt')])

    #Sort the index files by their number
    index_files = sorted(index_files, key=lambda f: int(f.split('_')[1].split('.')[0]))

    #Open all files and add the first line of each to a heap
    heap = []
    files = [open(os.path.join(directory, f)) for f in index_files]
    for i, file in enumerate(files):
        line = file.readline().strip()
        if line:
            processed = process_index_line(line)
            heap.append((processed[0], i, processed[1]))

    heapq.heapify(heap)
    current_token = '@'

    #Open the output file
    with open('merged_index.txt', 'w') as output_file:
        #Dictionary to keep track of the position of each token in the output file
        token_positions = {}
        current_position = 1
        #While there are still lines in the heap
        while heap:
            #Pop the next posting list off the heap and write it to the output file
            next_token, i, postings = heapq.heappop(heap)

            #Add to current token if the same
            if next_token == current_token:
                line = f',{postings}'
            #Otherwise, write a new line with the token and postings
            else:
                line = f'\n{next_token}:{postings}'
                current_token = next_token
                current_position += 1
                token_positions[next_token] = current_position

            current_position += len(line)
            output_file.write(line)

            #Add the next line from that file to the heap
            next_line = files[i].readline().strip()
            if next_line:
                processed = process_index_line(next_line)
                heapq.heappush(heap, (processed[0], i, processed[1]))

    #Close all the input files
    for file in files:
        file.close()

    with open(f'data/token_positions.pkl', 'wb') as f:
        pickle.dump(token_positions, f)  #Dump the token positions dictionary to disk


def build_index(folder_path, threshold):
    'Builds partial indexes and saves them to disk'
    inverted_index = defaultdict(list)
    page_counter = 0

    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')  #Matches any sequence of alphanum characters
    stemmer = Porter2Stemmer()

    universal_tokens = set()
    urls_processed = set()
    doc_ids = []

    for page in process_json_files(folder_path):
        
        #Don't process non html
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                tokens = tokenizer.tokenize(get_text_content(page))
            except:
                continue

        page_counter += 1
        url = remove_fragment(page['url'])
        if url in urls_processed:
            continue
        page_counter += 1
        doc_ids.append(url)
        urls_processed.add(url)

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

    with open(f'data/doc_ids.pkl', 'wb') as f:
        pickle.dump(doc_ids, f)  #Dump the doc_ids list to disk

    #Dump the remaining indexes
    if inverted_index:
        dump_indexes(f'indexes/index_{page_counter // threshold + 1}.txt', inverted_index)

    #ANALYTICS
    print(f'Total pages: {page_counter}')
    print(f'Total words: {len(universal_tokens)}')


if __name__ == '__main__':
    build_index('developer/DEV', 10000)
    merge_indexes('indexes')
