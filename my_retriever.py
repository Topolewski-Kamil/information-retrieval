import math
import re
import numpy as np

class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.number_of_docs()
        self.tfidfs = self.docs_tfidfs()
        self.tfs = self.docs_tfs()
        self.vectors = self.docs_vectors_len()
        self.bins = self.docs_bins()
    
    # Computer number of documents
    def number_of_docs(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    # Compute documents term frequencies for each term
    def docs_tfs(self):
        tfDict = {} # tf dict
       
        for term in self.index:
            for doc in self.index[term]:
                if doc not in tfDict:
                    tfDict[doc] = {}
                tfDict[doc][term] = self.index[term][doc]
        return tfDict

    # Compute documents binary weighting for each term
    def docs_bins(self):
        binDict = {} # tf dict
        for term in self.index:
            for doc in self.index[term]:
                if doc not in binDict:
                    binDict[doc] = {}
                binDict[doc][term] = 1
        return binDict

    # Compute documents tfidfs for each term
    def docs_tfidfs(self):
        tfidfsDict = {} # tf dict
        for term in self.index:
            count = len(self.index.get(term))
            idf = (math.log10(self.num_docs / count))
            for doc in self.index[term]:
                if doc not in tfidfsDict:
                    tfidfsDict[doc] = {}
                tf = self.index[term][doc]
                tfidfsDict[doc][term] = tf * idf
        return tfidfsDict
    
    # Compute documents length of vectors
    def docs_vectors_len(self):
        vectors = {}
        for doc in self.tfidfs:
            wordVec = np.array([])
            for term in self.tfidfs[doc]:
                wordVec = np.append(wordVec,self.tfidfs[doc][term])              
            vecLen = np.linalg.norm(wordVec)
            vectors[doc] = vecLen
        return vectors
    
    # Convert a query to a representation of dictionary of term counts
    def to_term_count(self, query):
        termDic = {}
        for word in query:
            if word not in termDic:
                termDic[word] = 1
            else:
                termDic[word] += 1
        return termDic
        
    # Find all documents with at least one word match in a query
    # and construct tf weight dict
    def relevant_docs_tf(self, query):
        documents = set() # relvant docs 
        tfDict = {} # tf dict
       
        for word in query:
            docIDs = self.index.get(word) # ids of all docs containg the word
            if docIDs != None:
                for docID in docIDs:
                    if docID not in tfDict:
                        tfDict[docID] = {}
                    tfDict[docID][word] = self.index[word][docID] 

                    documents.add(docID)        
        return documents, tfDict

    # Compute query term frequency
    def query_bin(self, query):
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
        query_terms = self.to_term_count(query)
        tfidf = {}
        for term in query:
            if self.index.get(term) != None:
                count = len(self.index.get(term))
                idf = (math.log10(self.num_docs / count))
                tfidf[term] = idf * query_terms[term]
        return tfidf    

    # Compute query vector length
    def query_vector(self, tfdif):
        wordVec = np.array([])
        for term in tfdif:
            wordVec = np.append(wordVec, tfdif[term])              
        vecLen = np.linalg.norm(wordVec)
        return vecLen

    # Compute cosine similarity
    def computing_cosine(self, tfidfQ, tfidfD):
        cosines = {}
        for doc in tfidfD:
            cosines[doc] = {}
            product = 0
            for term in tfidfQ:
                if term in tfidfD[doc]:
                    product += tfidfQ[term] * tfidfD[doc][term]
            cosines[doc] = product / self.vectors[doc]
        return cosines

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):

        if self.term_weighting == 'tfidf':
            tfdifQ = self.query_tfidf(query) # compute query tfidf
            self.query_vector(tfdifQ) # compute query vector length
            cosines = self.computing_cosine(tfdifQ, self.tfidfs)
        elif self.term_weighting == 'tf':
            tfQ = self.query_tf(query) # compute query term freq
            cosines = self.computing_cosine(tfQ, self.tfs)
        else:
            binsQ = self.query_bin(query) # compute query binary weights
            # print(binsQ)
            cosines = self.computing_cosine(binsQ, self.bins)
           

        cosines = dict(sorted(cosines.items(), key=lambda item: item[1], reverse=True))
        dict_items = cosines.items()
        chosen = list(dict_items)[:10] # get 10 best scoring

        finalList = []
        for tuple in chosen: #convert to a list
            finalList.append(tuple[0])
            

        return finalList


