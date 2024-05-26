import index
import time
import math

from collections import defaultdict
from posting import Posting
from porter2stemmer import Porter2Stemmer
from nltk.tokenize import RegexpTokenizer

def default_factory():
    return defaultdict(float)


def tfidf(term_freq, doc_freq, total_docs):
    'Calculates the tfidf score for a term'
    return (1 + math.log(term_freq)) * math.log(total_docs / (doc_freq + 1))

def normalize_tdif(doc_tdifs):
    if not doc_tdifs:
        return doc_tdifs
    
    min_tdif = min(doc_tdifs.values())
    max_tdif = max(doc_tdifs.values())
    for doc_id in doc_tdifs:
        doc_tdifs[doc_id] = (doc_tdifs[doc_id] - min_tdif) / (max_tdif - min_tdif)
    return doc_tdifs


def make_doc_vectors_and_tdif(postings, total_docs):
    'Creates a vector representation of the documents'
    doc_vectors = defaultdict(default_factory)
    doc_tfifs = defaultdict(float)
    for token, posting in postings.items():
        for post in posting:
            tfidf_value = tfidf(post.term_freq, post.doc_freq, total_docs)
            doc_vectors[post.doc_id][token] = tfidf_value
            doc_tfifs[post.doc_id] += tfidf_value 
    
    #Normalize the vectors
    for doc_id in doc_vectors:
        doc_vector = doc_vectors[doc_id]
        doc_length = math.sqrt(sum([value ** 2 for value in doc_vector.values()]))
        for token in doc_vector:
            doc_vector[token] /= doc_length
    
    return doc_vectors, doc_tfifs

def score(doc_cos, doc_tdifs, beta=0.15):
    'Scores the documents based on the cosine similarity and tdif'
    doc_tdifs = normalize_tdif(doc_tdifs)
    scores = defaultdict(float)
    for doc_id in doc_cos:
        scores[doc_id] = (beta * doc_tdifs[doc_id]) + ((1 - beta) * doc_cos[doc_id])
    
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
        query_vector[query_token] = tfidf(query_vector[query_token], doc_freq, len(doc_ids))

    #Normalize the query vector
    length = math.sqrt(sum(value**2 for value in query_vector.values()))

    for query_token in query_vector:
        query_vector[query_token] /= length
        

    #Get the postings for each token in the query
    postings = {}
    for token in query_vector.keys():
        postings[token] = index.get_postings(index_path, token, token_positions)

    doc_vectors, doc_tdifs = make_doc_vectors_and_tdif(postings, len(doc_ids))
    
    #Calculate the cosine similarity between the query and the documents
    doc_cos = defaultdict(float)
    for token in query_vector:
        for doc_id, doc_vector in doc_vectors.items():
            if token in doc_vector:
                doc_cos[doc_id] += (query_vector[token] * doc_vector[token])

    scores = score(doc_cos, doc_tdifs)

    #Rank postings
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)


    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Retreived results in {time_taken} seconds')
    return ([(doc_ids[ranked_docs[0]]) for doc in ranked_docs], time_taken)


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
    






