import index
import time
import math

from posting import Posting
from porter2stemmer import Porter2Stemmer
from nltk.tokenize import RegexpTokenizer

def merge_postings(postings1, postings2):
    'Merges two postings lists'
    merged_postings = []
    i = j = 0
    while i < len(postings1) and j < len(postings2):
        if postings1[i].doc_id == postings2[j].doc_id:
            merged_postings.extend([postings1[i], postings2[j]])
            i += 1
            j += 1
        elif postings1[i].doc_id < postings2[j].doc_id:
            i += 1
        else:
            j += 1
    return merged_postings

def merge_postings_list(postings_list):
    'Merges a list of postings lists'
    if not postings_list:
        return [] #Return empty list if no postings
    merged_postings = postings_list[0] #Start with the first postings list
    for postings in postings_list[1:]:
        merged_postings = merge_postings(merged_postings, postings) #Merge with the next postings list
    return merged_postings

def rank_documents(postings, total_docs):
    'Ranks the documents based on the postings list'
    doc_id_score = {}
    for posting in postings:
        #Add the tfidf score to the document
        doc_id_score[posting.doc_id] = (doc_id_score.get(posting.doc_id, 0) +
                                      tfidf(posting.term_freq, posting.doc_freq, total_docs))
    #Sort the documents by score
    postings = sorted(doc_id_score.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in postings]


def tfidf(term_freq, doc_freq, total_docs):
    'Calculates the tfidf score for a term'
    return (math.log(term_freq) + 1) * math.log(total_docs / doc_freq)


def search(query, index_path, token_positions, doc_ids):
    'Searches the index for the given query'
    start_time = time.time()
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')  #Matches any sequence of alphanum characters
    stemmer = Porter2Stemmer()

    #Tokenize and stem the query
    query_tokens = tokenizer.tokenize(query)
    query_tokens = [stemmer.stem(token.lower()) for token in query_tokens]

    #Get the postings for each token in the query
    postings = []
    for token in query_tokens:
        postings.append(index.get_postings(index_path, token, token_positions))
    
    #Combine postings for AND retreival
    postings = merge_postings_list(postings)
    #Rank postings
    postings = rank_documents(postings, len(doc_ids))

    end_time = time.time()
    print(f'Retreived results in {end_time - start_time} seconds')
    return [doc_ids[posting] for posting in postings]


def get_query(index_path, token_positions, doc_ids):
    'Gets a query from the user and prints search results'
    query =  input('Enter a search query: ')
    postings = search(query, index_path, token_positions, doc_ids)
    for i, posting in enumerate(postings[:10]):
        print(f'{i + 1} - {posting}')

if __name__ == '__main__':
    #Load the index and doc_ids
    token_positions = index.load_pickle_file('data/token_positions.pkl')
    doc_ids = index.load_pickle_file('data/doc_ids.pkl')

    get_query('merged_index.txt', token_positions, doc_ids)

    
    






