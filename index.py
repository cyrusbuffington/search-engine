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
from hashlib import sha256

def process_json_files(folder_path):
    'Yields data of each JSON file in folder_path'
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.json'):
                with open(os.path.join(root, filename), 'r') as f:
                    data = json.load(f)
                    yield data


def get_page_tokens(page, content_hashes):
    'Returns the text content of a JSON page object'
    tag_weights = {'title': 10, 'h1': 7, 'h2': 6, 'h3': 5, 'h4': 4, 'h5': 3, 'h6': 2, 'p': 1,
                    'a': 1, 'li': 1, 'i':3, 'b':4, 'strong': 4, 'em': 4, 'sub': 1}
    
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')  #Matches any sequence of alphanum characters
    stemmer = Porter2Stemmer()
    freqs = defaultdict(int)

    soup = BeautifulSoup(page['content'], 'lxml')
    content = ""

    for tag in soup.find_all(tag_weights.keys()):
        tag_content = tag.get_text()
        #Tokenize the tag content
        tokens = tokenizer.tokenize(tag_content)
        #Add the tag content to the page content
        content += tag_content
        for token in tokens:
            token = token.lower()
            stemmed_token = stemmer.stem(token)
            freqs[stemmed_token] += tag_weights[tag.name]

    #Hash the content to check for duplicates
    content_hash = sha256(content.encode()).hexdigest()

    

    if content_hash in content_hashes:
        print('Duplicate content detected')
        raise ValueError('Duplicate content')
    content_hashes.add(content_hash)

    return freqs



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
    print("Initializing index building...")
    inverted_index = defaultdict(list)
    page_counter = 0

    content_hashes = set()
    universal_tokens = set()
    urls_processed = set()
    doc_ids = []


    for page in process_json_files(folder_path):
        freqs = defaultdict(int)

        url = remove_fragment(page['url'])
        if url in urls_processed:
            continue
        
        #Don't process non html or broken html or duplicate content
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                freqs = get_page_tokens(page, content_hashes)
            except:
                continue
        
        page_counter += 1
        doc_ids.append(url)
        urls_processed.add(url)


        for token, freq in freqs.items():
            inverted_index[token].append(Posting(page_counter - 1, freq)) #Add to inverted index
            universal_tokens.add(token)

        #Dump indexes
        if page_counter % threshold == 0:
            print('Dumping indexes... - ', page_counter // threshold)
            dump_indexes(f'indexes/index_{page_counter // threshold}.txt', inverted_index)
            inverted_index.clear()

    with open(f'data/doc_ids.pkl', 'wb') as f:
        pickle.dump(doc_ids, f)  #Dump the doc_ids list to disk

    #Dump the remaining indexes
    if inverted_index:
        print('Dumping indexes... - ', page_counter // threshold + 1)
        dump_indexes(f'indexes/index_{page_counter // threshold + 1}.txt', inverted_index)

    print('Index building complete!')
    #ANALYTICS
    print(f'Total pages: {page_counter}')
    print(f'Total words: {len(universal_tokens)}')


def main():
    build_index('developer/DEV', 5000)
    merge_indexes('indexes')


if __name__ == '__main__':
    main()