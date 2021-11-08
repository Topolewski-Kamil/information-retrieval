import math
import numpy as np

class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.number_of_docs()
        self.SMOOTING_TERM = 0.4

        if (term_weighting == 'tfidf'):
            self.weights = self.docs_tfidfs()
        elif (term_weighting == 'tf'):
            self.weights = self.docs_tfs()
        else:
            self.weights = self.docs_binary()
        self.vectors = self.docs_vectors_len()
    
    # Compute number of documents
    def number_of_docs(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    # Find all documents with at least one word match in a query
    # and construct tf weight dict
    def relevant_docs_tf(self, query):
        documents = set() # relvant docs        
        for word in query:
            doc_ids = self.index.get(word) # ids of all docs containg the word
            if doc_ids != None:
                for doc_id in doc_ids:
                    documents.add(doc_id)        
        return documents

    # Compute documents binary weighting for each term
    def docs_binary(self):
        bin_dict = {} # tf dict
        for term in self.index:
            for doc in self.index[term]:
                if doc not in bin_dict:
                    bin_dict[doc] = {}
                bin_dict[doc][term] = 1
        return bin_dict

    # Find max term occurances for each document
    def max_term_frequency(self, tf):
        tf_max = {}
        for doc in tf:
            tf[doc] = dict(sorted(tf[doc].items(), key=lambda item: item[1], reverse=True)) # descending order
            highestTf = list(tf[doc].values())[0]
            tf_max[doc] = highestTf
        return tf_max

    # Compute documents term frequencies for each document
    def docs_max_tfs(self, tf, tf_max):
        for doc in tf:
            for term in tf[doc]:
                tf[doc][term] = self.SMOOTING_TERM + ((1 - self.SMOOTING_TERM) * (tf[doc][term]/tf_max[doc]))
        return tf

    # Compute documents log term frequencies for each term
    def docs_tfs(self):
        tf_dict = {} # tf dict
        for term in self.index:
            for doc in self.index[term]:
                if doc not in tf_dict:
                    tf_dict[doc] = {}
                if self.index[term][doc] > 0:
                    tf_dict[doc][term] = 1 + (math.log10(self.index[term][doc]))
                else: 
                    tf_dict[doc][term] = 0
        return tf_dict

    # Compute documents tfidfs for each term 
    def docs_tfidfs(self):
        tfidfs_dict = {} # tf dict
        for term in self.index:
            count = len(self.index.get(term))
            idf = (math.log10(self.num_docs / count))
            for doc in self.index[term]:
                if doc not in tfidfs_dict:
                    tfidfs_dict[doc] = {}
                tf = self.index[term][doc]
                tfidfs_dict[doc][term] = tf * idf
        return tfidfs_dict
    
    # Compute documents length of vectors
    def docs_vectors_len(self):
        vectors = {}
        for doc in self.weights:
            word_vec = np.array([])
            for term in self.weights[doc]:
                word_vec = np.append(word_vec,self.weights[doc][term])              
            vectors[doc] = np.linalg.norm(word_vec)
        return vectors

    # Compute query binary term frequency
    def query_binary(self, query):
        tf = {}
        for term in query:
            if term not in tf:
                tf[term] = 1
        return tf

    # Compute query term frequency
    def query_tf(self, query):
        tf = {}
        for term in query:
            if term in tf:
                tf[term] += 1
            else:
                tf[term] = 1
        return tf

    # Compute query tfidf
    def query_tfidf(self, query):
        query_terms = self.query_tf(query)
        tfidf = {}
        for term in query:
            if self.index.get(term) != None:
                count = len(self.index.get(term))
                idf = (math.log10(self.num_docs / count))
                tfidf[term] = idf * query_terms[term]
        return tfidf  

    # Compute query vector length (not neccessary in this assignment)
    def query_vector(self, tfdif):
        word_vec = np.array([])
        for term in tfdif:
            word_vec = np.append(word_vec, tfdif[term])              
        vec_len = np.linalg.norm(word_vec)
        return vec_len

    # Compute cosine similarity
    def computing_cosine(self, tfidf_query, tfidf_docs, relevant_docs):
        cosines = {}
        for doc in tfidf_docs:
            if doc in relevant_docs:
                cosines[doc] = {}
                product = 0
                for term in tfidf_query:
                    if term in tfidf_docs[doc]:
                        product += tfidf_query[term] * tfidf_docs[doc][term]
                cosines[doc] = product / self.vectors[doc]
        return cosines

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        relevant_docs_ids = self.relevant_docs_tf(query)

        if self.term_weighting == 'tfidf':
            weights_query = self.query_tfidf(query) # compute query tfidf weights
        elif self.term_weighting == 'tf':
            weights_query = self.query_tf(query) # compute query term freq weights
        else:
            weights_query = self.query_binary(query) # compute query binary weights
        
        cosines = self.computing_cosine(weights_query, self.weights, relevant_docs_ids)
        cosines = dict(sorted(cosines.items(), key=lambda item: item[1], reverse=True)) # descending order
        cosines_items = cosines.items()
        top_cosines = list(cosines_items)[:10] # get 10 best scoring

        chosen_docs = []
        for tuple in top_cosines: #convert to a list of only ids
            chosen_docs.append(tuple[0])
            
        return chosen_docs