import math
import numpy as np

class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.number_of_docs()

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

    # Compute documents binary weighting for each term
    def docs_binary(self):
        binDict = {} # tf dict
        for term in self.index:
            for doc in self.index[term]:
                if doc not in binDict:
                    binDict[doc] = {}
                binDict[doc][term] = 1
        return binDict

    # Compute documents term frequencies for each term
    def docs_tfs(self):
        tfDict = {} # tf dict
        for term in self.index:
            for doc in self.index[term]:
                if doc not in tfDict:
                    tfDict[doc] = {}
                if self.index[term][doc] > 0:
                    tfDict[doc][term] = 1 + (math.log10(self.index[term][doc]))
                else: 
                    tfDict[doc][term] = 0
        return tfDict

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
        for doc in self.weights:
            wordVec = np.array([])
            for term in self.weights[doc]:
                wordVec = np.append(wordVec,self.weights[doc][term])              
            vecLen = np.linalg.norm(wordVec)
            vectors[doc] = vecLen
        return vectors
        
    # Find all documents with at least one word match in a query
    # and construct tf weight dict
    def relevant_docs_tf(self, query):
        documents = set() # relvant docs        
        for word in query:
            docIDs = self.index.get(word) # ids of all docs containg the word
            if docIDs != None:
                for docID in docIDs:
                    documents.add(docID)        
        return documents

    # Compute query term frequency
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

    # Compute log weight of query
    def logTfweight(self, tf):
        for term in tf:
            if tf[term] > 0:
                tf[term] = 1 + (math.log10(tf[term]))
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

    # Compute query vector length
    def query_vector(self, tfdif):
        wordVec = np.array([])
        for term in tfdif:
            wordVec = np.append(wordVec, tfdif[term])              
        vecLen = np.linalg.norm(wordVec)
        return vecLen

    # Compute cosine similarity
    def computing_cosine(self, tfidfQ, tfidfD, relevantDocs):
        cosines = {}
        for doc in tfidfD:
            if doc in relevantDocs:
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
        relevantDocsIDs = self.relevant_docs_tf(query)

        if self.term_weighting == 'tfidf':
            weightsQ = self.query_tfidf(query) # compute query tfidf weights
        elif self.term_weighting == 'tf':
            weightsQ = self.query_tf(query) # compute query term freq weights
        else:
            weightsQ = self.query_binary(query) # compute query binary weights
        
        cosines = self.computing_cosine(weightsQ, self.weights, relevantDocsIDs)
        cosines = dict(sorted(cosines.items(), key=lambda item: item[1], reverse=True)) # descending order
        cosines_items = cosines.items()
        top10cosines = list(cosines_items)[:10] # get 10 best scoring

        chosenDocuments = []
        for tuple in top10cosines: #convert to a list of only ids
            chosenDocuments.append(tuple[0])
            
        return chosenDocuments