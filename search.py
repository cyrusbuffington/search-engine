import index
import time
import math
import heapq

from collections import defaultdict
from posting import Posting
from porter2stemmer import Porter2Stemmer
from nltk.tokenize import RegexpTokenizer



def default_factory():
    return defaultdict(float)


def tfidf(term_freq, doc_freq, total_docs):
    'Calculates the tfidf score for a term'
    return (1 + math.log(term_freq)) * math.log(total_docs / (doc_freq + 1))

def normalize_tfidfs(doc_tfidfs):
    if not doc_tfidfs:
        return doc_tfidfs
    
    min_tdif = min(doc_tfidfs.values())
    max_tdif = max(doc_tfidfs.values())
    for doc_id in doc_tfidfs:
        doc_tfidfs[doc_id] = (doc_tfidfs[doc_id] - min_tdif) / (max_tdif - min_tdif)
    return doc_tfidfs


def make_doc_vectors_and_tdif(postings, total_docs):
    'Creates a vector representation of the documents'
    doc_vectors = defaultdict(default_factory)
    doc_tfidfs = defaultdict(float)
    for token, posting in postings.items():
        for post in posting:
            tfidf_value = tfidf(post.term_freq, post.doc_freq, total_docs)
            doc_vectors[post.doc_id][token] = tfidf_value
            doc_tfidfs[post.doc_id] += tfidf_value 
    
    #Normalize the vectors
    for doc_id in doc_vectors:
        doc_vector = doc_vectors[doc_id]
        doc_length = math.sqrt(sum([value ** 2 for value in doc_vector.values()]))
        for token in doc_vector:
            doc_vector[token] /= doc_length
    
    return doc_vectors, doc_tfidfs

def heap_selection(doc_vectors, query_vector, k=125):
    'Selects the top k documents using a heap'
    heap = []
    for doc_id, doc_vector in doc_vectors.items():
        cos_sim = 0
        for token in query_vector:
            if token in doc_vector:
                cos_sim += (doc_vector[token] * query_vector[token])
        if len(heap) < k:
            heapq.heappush(heap, (cos_sim, doc_id))
        elif cos_sim > heap[0][0]:
            heapq.heappushpop(heap, (cos_sim, doc_id))

    doc_cos = defaultdict(float)
    for i in range(len(heap)):
        doc_cos[heap[i][1]] = heap[i][0]
    return doc_cos


def score(doc_cos, doc_tfidfs, beta=.15):
    'Scores the documents based on the cosine similarity and tdif'
    doc_tfidfs = normalize_tfidfs(doc_tfidfs)
    scores = defaultdict(float)
    for doc_id in doc_cos:
        scores[doc_id] = (beta * doc_tfidfs[doc_id]) + ((1 - beta) * doc_cos[doc_id])
    
    return scores
            

def search(query, index_path, token_positions, doc_ids):
    'Searches the index for the given query'
    if not query:
        return []
    start_time = time.time()
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')  #Matches any sequence of alphanum characters
    stemmer = Porter2Stemmer()

    #Tokenize and stem the query
    query_tokens = tokenizer.tokenize(query)
    query_vector = {}
    for token in query_tokens:
        token = token.lower()
        stemmed_token = stemmer.stem(token)
        query_vector[stemmed_token] = query_vector.get(stemmed_token, 0) + 1

    #Calculate the tfidf score for the query
    for query_token in query_vector:
        doc_freq = len(index.get_postings(index_path, query_token, token_positions))
        token_tfidf = tfidf(query_vector[query_token], doc_freq, len(doc_ids))
        query_vector[query_token] = token_tfidf


    #Normalize the query vector
    length = math.sqrt(sum(value**2 for value in query_vector.values()))

    for query_token in query_vector:
        query_vector[query_token] /= length
        

    #Get the postings for each token in the query
    postings = {}
    for token in query_vector.keys():
        postings[token] = index.get_postings(index_path, token, token_positions)

    doc_vectors, doc_tfidfs = make_doc_vectors_and_tdif(postings, len(doc_ids))


    #Calculate the cosine similarity between the query and the documents
    doc_cos = heap_selection(doc_vectors, query_vector)

    scores = score(doc_cos, doc_tfidfs)

    #Rank postings
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Retreived results in {time_taken} seconds')
    return ([(doc_ids[doc[0]]) for doc in ranked_docs], time_taken)


def get_query(index_path, token_positions, doc_ids):
    'Gets a query from the user and prints search results'
    query =  input('Enter a search query: ')
    postings = search(query, index_path, token_positions, doc_ids)[0]
    for i, posting in enumerate(postings[:40]):
        print(f'{i + 1} - {posting}')

def main():
    #Load the index and doc_ids
    token_positions = index.load_pickle_file('data/token_positions.pkl')
    doc_ids = index.load_pickle_file('data/doc_ids.pkl')

    get_query('merged_index.txt', token_positions, doc_ids)


if __name__ == '__main__':
    main()
    






