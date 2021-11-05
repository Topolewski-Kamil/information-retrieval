import sys, re, getopt, glob, math
from collections import Counter
from types import WrapperDescriptorType

class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        # self.idfs = self.compute_idfs()
        # self.tfs = self.compute_tfs()
        self.tfidfs = self.compute_tfidfs()
        
    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    # Compute tf for all terms
    def compute_tfs(self):
        tfDict = {} # tf dict
       
        for term in self.index:
            for doc in self.index[term]:
                if doc not in tfDict:
                    tfDict[doc] = {}
                tfDict[doc][term] = self.index[term][doc]
        return tfDict

    # Compute idfs for all terms
    def compute_idfs(self):
        idfs = {}
        for term in self.index:
            idfs[term] = {}
            docIDs = self.index.get(term) # ids of all docs containg the word
            idf = (math.log10(self.num_docs / len(docIDs)))
            idfs[term] = idf 
            
        return idfs

    def compute_tfidfs(self):
        tfidfsDict = {} # tf dict
        for term in self.index:
            count = len(self.index.get(term))
            idf = (math.log10(self.num_docs / count))
            for doc in self.index[term]:
                if doc not in tfidfsDict:
                    tfidfsDict[doc] = {}
                
                tf = self.index[term][doc]
                tfidfsDict[doc][term] = tf * idf

        print(tfidfsDict[2931])
        return tfidfsDict
    
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
        queryTermCount = self.to_term_count(query)
       
        for word in query:
            docIDs = self.index.get(word) # ids of all docs containg the word
            if docIDs != None:
                for docID in docIDs:
                    if docID not in tfDict:
                        tfDict[docID] = {}
                    tfDict[docID][word] = self.index[word][docID] 

                    documents.add(docID)        
        return documents, tfDict


    # Find all documents with at least one word match in a query
    # and construct tfidf dict
    def relevant_docs_tfidf(self, query):
        documents = set() # relvant docs 
        tfidfDict = {} # tfidf dict
        tfidfQuery = {} # tfidf for query 
        queryTermCount = self.to_term_count(query)

        for word in query:               
            docIDs = self.index.get(word) # ids of all docs containg the word
            if docIDs != None:
                idf = (math.log10(self.num_docs / len(docIDs)))
                tfidfQuery[word] = idf * queryTermCount[word]
                for docID in docIDs:
                    if docID not in tfidfDict:
                        tfidfDict[docID] = {}
                    tfidfDict[docID][word] =  idf * self.index[word][docID]
            
        return documents, tfidfDict, tfidfQuery


    def compute_vectors(self, docsIDs):
        output_dict = {}
        for val,item in enumerate(docsIDs):
            output_dict[item] = 0

        for term in self.index:
            for docID in self.index[term]:
                output_dict

        return output_dict




    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        docsIDs, tfDic = self.relevant_docs_tf(query)
        docsIDs2 , tfidfDic, tfidfQuery = self.relevant_docs_tfidf(query)
        # self.compute_vectors(docsIDs)
        # if query[0] == 'what' and query[1] == 'articles':
            # print(self.tfidfs)      
        
        return list(docsIDs)


