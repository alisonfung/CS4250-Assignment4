#-------------------------------------------------------------------------
# AUTHOR: Alison Fung
# FILENAME: main.py
# SPECIFICATION: Creates an inverted index based on documents and scores
#               the documents with given queries using cosine similarity
#               with tf-idf weights.
# FOR: CS 4250- Assignment #4
# TIME SPENT: 8 hours
#-----------------------------------------------------------*/

import numpy as np
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer


# calculate tf-idf weights for documents and queries
def create_tfidf_vectors(documents, queries):
    # use 1-3 grams to create terms
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3))
    # calculate document tf-idf weights
    term_vectors = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()
    # calculate query tf-idf weights
    query_vectors = vectorizer.transform(queries)
    return term_vectors, terms, query_vectors


# connect to the database
def connect_database():
    client = MongoClient(host="localhost", port=27017)
    db = client.assignment4
    return db


# store term in inverted index
def store_term(_id, pos, docs, col):
    col.update_one({"_id": _id}, {"$set": {
        "pos": pos,
        "docs": docs
    }}, True)


# get term from inverted index
def get_term(pos, col):
    return col.find_one({"pos": int(pos)})


# store document in database
def store_document(_id, content, col):
    col.update_one({"_id": _id}, {"$set": {
        "content": content,
    }}, True)


# get document from database
def get_document(_id, col):
    return col.find_one({"_id": _id})


def main():
    # connect to database and set up collections
    db = connect_database()
    term_col = db["terms"]
    doc_col = db["documents"]

    # document text
    doc1 = "After the medication, headache and nausea were reported by the patient."
    doc2 = "The patient reported nausea and dizziness caused by the medication."
    doc3 = "Headache and dizziness are common effects of this medication."
    doc4 = "The medication caused a headache and nausea, but no dizziness was reported."
    documents = [doc1, doc2, doc3, doc4]

    # query text
    q1 = "nausea and dizziness"
    q2 = "effects"
    q3 = "nausea was reported"
    q4 = "dizziness"
    q5 = "the medication"
    queries = [q1, q2, q3, q4, q5]

    # store documents in database
    for i in range(len(documents)):
        store_document(i, documents[i], doc_col)

    # create tf-idf weight vectors
    term_matrix, terms, query_matrix = create_tfidf_vectors(documents, queries)
    terms = terms.tolist()

    # create the inverted index
    # for each term
    for term in range(len(terms)):
        docs = []
        # for each document
        for document in range(len(documents)):
            # if the term weight for the document is nonzero
            if term_matrix[document, term] != 0:
                # add it to the list of documents
                docs.append({"doc_num": document, "weight": term_matrix[document, term]})
        # sort documents by weight descending
        docs.sort(key=lambda d: d["weight"], reverse=True)
        # store term in index
        store_term(term, term, docs, term_col)

    # calculate query scores
    # for each query
    for i in range(len(queries)):
        query_vector = query_matrix[i]
        # get the corresponding vector's nonzero positions
        query_vector_nonzero = query_vector.nonzero() # (row, col)
        # initialize a list of weights for each document
        matched_document_weights = [[] for _ in range(len(documents))]
        # list of matched documents with scores
        cos_sim_documents = []

        # for each term in the query
        for j in range(len(query_vector_nonzero[1])):
            # get the entry in the inverted index for this term
            index_data = get_term(query_vector_nonzero[1][j], term_col)
            term_pos = index_data["pos"]
            term_docs = index_data["docs"]
            # for each document listed for this term
            for k in range(len(term_docs)):
                # get the document number
                doc_num = term_docs[k]["doc_num"]
                # get the document weight
                doc_weight = term_docs[k]["weight"]
                # add it to the list for the corresponding document
                matched_document_weights[doc_num].append({"weight": doc_weight, "pos": term_pos})

        # for each document
        for j in range(len(matched_document_weights)):
            # if the document had no matched terms, skip it
            if len(matched_document_weights[j]) == 0:
                continue
            # initialize dot product (cosine similarity), values are already normalized from TfidfVectorizer
            dot_product = 0
            # for each matched term in this document
            for k in range(len(matched_document_weights[j])):
                # get the weight
                document_weight = matched_document_weights[j][k]["weight"]
                # if the weight is nonzero
                if document_weight != 0:
                    # get the position
                    document_term_pos = matched_document_weights[j][k]["pos"]
                    # get the corresponding query term weight
                    query_weight = query_vector[0, document_term_pos]
                    # add this term to the dot product
                    dot_product += document_weight * query_weight
            # add the dot product to the result list for this query
            cos_sim_documents.append({"doc_num": j, "score": dot_product})

        # sort the matched documents by score descending
        cos_sim_documents.sort(key=lambda d: d["score"], reverse=True)

        # print output
        # print query text
        print("Query:", queries[i])
        # print matched document content and score rounded to 2 decimal places
        for j in range(len(cos_sim_documents)):
            document_content = get_document(cos_sim_documents[j]["doc_num"], doc_col)["content"]
            print(document_content, np.round(cos_sim_documents[j]["score"], 2))
        print()


if __name__ == '__main__':
    main()
