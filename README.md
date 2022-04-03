I have built an IR system that performs retrieval based on the vector space model with alternative term weighting schemes. The system has been tested over the CACM test collection.

To be able to construct a vector space model, terms in the dictionary have to be weighted. Dependent on the weighting scheme, different variables must be present. For Binary: binary weights, TF: term weights, TFIDS: term weights and inverse document frequencies. All of these variables do not depend on queries, therefore they can be computed only once, which is done in the init function.

I have also tested two additional schemes for term weighting: logarithmic and maximum tf normalization. For those, I only had to adjust the weights calculated.

After calculating a weight according to the weighting scheme, a vector length is computed. Each document has a vector length and is stored in a dictionary {docID ⇒ vector length}. For each query, weight (based on weighting scheme) is computed and then a vector length (I have written the function, although it is unnecessary for this assignment). Finally, cosine similarity between the query and each document is calculated. Before actual computation, I discard all irrelevant documents (no word matches with the query).

In the end, we pick ten best fitting documents based on the highest cosine similarity and return
them.
