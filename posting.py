class Posting:
    def __init__(self, doc_id, term_freq, doc_freq=None):
        self.doc_id = doc_id
        self.term_freq = term_freq
        self.doc_freq = doc_freq
        
