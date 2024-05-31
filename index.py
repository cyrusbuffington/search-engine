import os
import json
import pickle
import heapq
import warnings
import time
from urllib.parse import urljoin
from posting import Posting
from porter2stemmer import Porter2Stemmer
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup
from collections import defaultdict
from operator import attrgetter
from hashlib import sha256
import unittest

def process_json_files(folder_path):
    'Yields data of each JSON file in folder_path'
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.json'):
                with open(os.path.join(root, filename), 'r') as f:
                    data = json.load(f)
                    yield data


def get_root_url(url):
    'Returns the root URL of a given URL'
    return url.split('/')[2]

def is_root_url(url):
    'Returns True if the URL is a root URL'
    return len(url.split('/')) == 3

def hash_page_content(content):
    'Hashes the content of a page to check for duplicates'
    return sha256(content.encode()).hexdigest()

def simhash(content_freq):
    'Calculates simhash of page'
    #Make 32 bit vector
    v = [0] * 32
    #Hash each token
    #If hash bit is 1, add weight to vector, otherwise subtract
    for token, weight in content_freq.items():
        h = hash(token) & 0xffffffff
        h = format(h, '032b')
        for i, bit in enumerate(h):
            if bit == '1':
                v[i] += weight
            else:
                v[i] -= weight
    #Create fingerprint
    fingerprint = 0
    
    for i in range(32):
        if v[i] > 0:
            fingerprint |= (1 << i)

    return fingerprint
    

def similarity(fingerprint1, fingerprint2):
    same_bits = 0
    for i in range(32):
        if fingerprint1 & (1 << i) == fingerprint2 & (1 << i):
            same_bits += 1
    return same_bits / 32


def get_page_tokens(soup, content_hashes, url):
    'Returns the text content of a JSON page object'
    tag_weights = {'title': 10, 'h1': 7, 'h2': 6, 'h3': 5, 'h4': 4, 'h5': 3, 'h6': 2, 'p': 1,
                    'a': 1, 'li': 1, 'i':3, 'b':4, 'strong': 4, 'em': 4, 'sub': 1}
    
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')  #Matches any sequence of alphanum characters
    stemmer = Porter2Stemmer()
    freqs = defaultdict(int)

    content_tokens = []
    for tag in soup.find_all(tag_weights.keys()):
        tag_content = tag.get_text()
        #Tokenize the tag content
        tokens = tokenizer.tokenize(tag_content)
        #Add the tag content to the page content
        content_tokens.extend(tokens)
        for token in tokens:
            token = token.lower()
            stemmed_token = stemmer.stem(token)
            #Add the token to the frequency dictionary based on term weight
            freqs[stemmed_token] += tag_weights[tag.name]

    #Hash the content to check for duplicates
    content_hash = simhash(freqs)

    root_url = get_root_url(url)

    if not is_root_url(url) and root_url in content_hashes:
        for hash in content_hashes[root_url]:
            if similarity(content_hash, hash) >= .97:
                print(url)
                raise ValueError('Near duplicate content found')

    content_hashes[root_url].add(content_hash)

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
        #Get the postings list from the line
        postings = process_index_line(line)[1].split(',')
        #Create a list of Posting objects from the postings list
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


def relative_to_abolute_url(url, link):
    'Converts a relative URL to an absolute URL'
    if link.startswith('//'):
        link = 'http:' + link
    elif link.startswith('/'):
        link = urljoin(url, link)
    return link


def build_pagerank_graph_with_ids(url_to_ids, pagerank_graph):
    'Builds the pagerank graph from the url_to_ids and pagerank_graph dictionaries'
    outgoing_links = defaultdict(int)
    graph = defaultdict(list)
    for url, linked_urls in pagerank_graph.items():
        if url not in url_to_ids:
            continue
        for linked_url in linked_urls:
            if linked_url not in url_to_ids:
                continue

            graph[url_to_ids[url]].append(url_to_ids[linked_url])
        outgoing_links[url_to_ids[url]] = len(linked_urls)

    reverse_graph = defaultdict(list)
    #Make graph of all pages that link to a page
    for id, linked_ids in graph.items():
        for linked_id in linked_ids:
            reverse_graph[linked_id].append(id)
    
    return reverse_graph, outgoing_links


def calculate_pagerank(reverse_graph, outgoing_links, pagerank, d=0.85, max_iter=1000):
    'Calculates the pagerank of each page in the graph'
    N = len(pagerank)
    for _ in range(max_iter):
        for i in range(N):
            sum_pr = 0
            for j in reverse_graph[i]:
                sum_pr += pagerank[j] / outgoing_links[j]
            pagerank[i] = (1 - d)  + d * sum_pr

def normalize_pagerank(pagerank):
    'Normalizes the pagerank values'
    max_pr = max(pagerank)
    min_pr = min(pagerank)
    for i in range(len(pagerank)):
        pagerank[i] = (pagerank[i] - min_pr) / (max_pr - min_pr)
    return pagerank

def build_index(folder_path, threshold):
    'Builds partial indexes and saves them to disk'
    start = time.perf_counter()
    print("Initializing index building...")
    inverted_index = defaultdict(list)
    page_counter = 0

    content_hashes = defaultdict(set)
    universal_tokens = set()
    doc_ids = []

    #Pagerank attributes
    url_to_ids = {}
    pagerank = []
    pagerank_graph = defaultdict(list)


    for page in process_json_files(folder_path):
        freqs = defaultdict(int)

        #Remove fragment from URL
        url = remove_fragment(page['url'])
        if url in url_to_ids:
            continue
        
        #Don't process non html or broken html or duplicate content
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                soup = BeautifulSoup(page['content'], 'lxml')
                freqs = get_page_tokens(soup, content_hashes, url)

            except:
                continue
        if page_counter % 100 == 0:
            print(f'Processed {page_counter} pages...')
        #Build sparse graph for pagerank
        links = soup.find_all('a')
        for link in links:
            link = link.get('href')
            if not link:
                continue
            link = link.strip()

            linked_url = remove_fragment(link)

            #Make relative url into absolute url
            linked_url = relative_to_abolute_url(url, linked_url)

            pagerank_graph[url].append(linked_url)

        
        #Add the page to the doc_ids list
        url_to_ids[url] = page_counter
        page_counter += 1
        #Add the page to the doc_ids list
        doc_ids.append(url)
        #Add the pagerank to the pagerank list
        pagerank.append(1)


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

    #Build pagerank graph
    print('Building pagerank graph...')
    reverse_graph, outgoing_links = build_pagerank_graph_with_ids(url_to_ids, pagerank_graph)

#DUMP DATA TO CALCULATE RAGERANK LATER
    with open(f'data/reverse_graph.pkl', 'wb') as f:
        pickle.dump(reverse_graph, f)
    with open(f'data/outgoing_links.pkl', 'wb') as f:
        pickle.dump(outgoing_links, f)

    #Calculate pagerank
    print('Calculating pagerank...')
    calculate_pagerank(reverse_graph, outgoing_links, pagerank)
    pagerank = normalize_pagerank(pagerank)
    #Dump pagerank
    print('Dumping pagerank...')
    with open(f'data/pagerank.pkl', 'wb') as f:
        pickle.dump(pagerank, f)  #Dump the pagerank list to disk
    print('Pagerank complete!')

    end = time.perf_counter()

    print('Index building took ', end - start, ' seconds')



def main():
    build_index('developer/DEV/', 5000)
    merge_indexes('indexes')


class TestSimHash(unittest.TestCase):
    def test_simhash(self):
        content_freq = {"hello": 1, "world": 2}
        result = simhash(content_freq)
        self.assertIsInstance(result, int)

    def test_similarity(self):
        fingerprint1 = 0b10101010101010101010101010101010
        fingerprint2 = 0b10101010101010101010101010101010
        result = similarity(fingerprint1, fingerprint2)
        self.assertEqual(result, 1.0)

        fingerprint1 = 0b10101010101010101010101010101010
        fingerprint2 = 0b01010101010101010101010101010101
        result = similarity(fingerprint1, fingerprint2)
        self.assertEqual(result, 0.0)


if __name__ == '__main__':
    #unittest.main()
    main()