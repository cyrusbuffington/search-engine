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


def make_doc_vectors_and_tdif(postings, total_docs, top_docs):
    'Creates a vector representation of the documents'
    doc_vectors = defaultdict(default_factory)
    doc_tfidfs = defaultdict(float)
    for token, posting in postings.items():
        for post in posting:
            if post.doc_id not in top_docs:
                continue
            tfidf_value = tfidf(post.term_freq, post.doc_freq, total_docs)
            #Add the tfidf value to the document vector
            doc_vectors[post.doc_id][token] = tfidf_value
            #Sum the tdif values for each document
            doc_tfidfs[post.doc_id] += tfidf_value 
    
    #Normalize the vectors
    for doc_id in doc_vectors:
        doc_vector = doc_vectors[doc_id]
        doc_length = math.sqrt(sum([value ** 2 for value in doc_vector.values()]))
        for token in doc_vector:
            doc_vector[token] /= doc_length
    
    return doc_vectors, doc_tfidfs


def cos_heap_selection(doc_vectors, query_vector, k=100):
    'Selects the top k documents using a heap'
    heap = []
    for doc_id, doc_vector in doc_vectors.items():
        cos_sim = 0
        #Get the cosine similarity between the query and the document
        for token in query_vector:
            if token in doc_vector:
                cos_sim += (doc_vector[token] * query_vector[token])
        #Only add the document to the heap if it has a higher cosine similarity than the smallest value in the heap
        if len(heap) < k:
            heapq.heappush(heap, (cos_sim, doc_id))
        elif cos_sim > heap[0][0]:
            heapq.heappushpop(heap, (cos_sim, doc_id))

    #Convert the heap to a dictionary for easier access
    doc_cos = defaultdict(float)
    for i in range(len(heap)):
        doc_cos[heap[i][1]] = heap[i][0]
    return doc_cos

def pagerank_heap_selection(postings, pagerank, k=5000):
    'Selects the top k documents based on pagerankusing a heap'
    top_docs = set()
    heap = []
    for token, posting in postings.items():
        for post in posting:
            doc_id = post.doc_id
            if doc_id not in top_docs:
                if len(heap) < k:
                    heapq.heappush(heap, (pagerank[doc_id], doc_id))
                elif pagerank[doc_id] > heap[0][0]:
                    heapq.heappushpop(heap, (pagerank[doc_id], doc_id))
                
    for i in range(len(heap)):
        top_docs.add(heap[i][1])

    return top_docs


def score(doc_cos, doc_tfidfs, pagerank, beta=.15, alpha = .9):
    'Scores the documents based on the cosine similarity and tdif'
    doc_tfidfs = normalize_tfidfs(doc_tfidfs)
    scores = defaultdict(float)
    #Compute ranking function for each doc
    for doc_id in doc_cos:
        scores[doc_id] = alpha * ((beta * doc_tfidfs[doc_id]) + ((1 - beta) * doc_cos[doc_id])) + (1 - alpha) * pagerank[doc_id]

    return scores
            

def search(query, index_path, token_positions, doc_ids, pagerank):
    'Searches the index for the given query'
    if not query:
        return []
    start_time = time.perf_counter()
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

    top_docs = pagerank_heap_selection(postings, pagerank)
    #Create document vectors and tdif scores
    doc_vectors, doc_tfidfs = make_doc_vectors_and_tdif(postings, len(doc_ids), top_docs)

    #Calculate the top k cosine similarities between the query and the documents
    doc_cos = cos_heap_selection(doc_vectors, query_vector)

    #Score the documents
    scores = score(doc_cos, doc_tfidfs, pagerank)

    #Rank postings
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    end_time = time.perf_counter()
    time_taken = end_time - start_time
    print(f'Retreived results in {time_taken} seconds')
    return ([(doc_ids[doc[0]]) for doc in ranked_docs], time_taken)


def get_query(index_path, token_positions, doc_ids, pagerank):
    'Gets a query from the user and prints search results'
    query =  input('Enter a search query: ')
    postings = search(query, index_path, token_positions, doc_ids, pagerank)[0]
    for i, posting in enumerate(postings[:10]):
        print(f'{i + 1} - {posting}')


def main():
    token_positions = index.load_pickle_file('data/token_positions.pkl')
    doc_ids = index.load_pickle_file('data/doc_ids.pkl')
    pagerank = index.load_pickle_file('data/pagerank.pkl')

    get_query('merged_index.txt', token_positions, doc_ids, pagerank)

    


if __name__ == '__main__':
    main()
    






