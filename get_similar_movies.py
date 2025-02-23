import nltk
import pickle
import numpy as np
import pandas as pd
import sys
import json
import warnings

from functools import wraps
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import *
warnings.filterwarnings("ignore")

example_sent = sys.argv[1]
json_path = sys.argv[2]

with open(json_path, "r") as f:
    config = json.load(f)


def create_tfidf_embedding(config):

    """Read and preprocess text from dataframe and use it down the line for TFIDF vectorization."""

    # load the dataset
    df = load_data(config)
    # process the text
    reqd_df = process_dataframe(df, config)
    
    # initialize the stemming process
    porter_stemmer = PorterStemmer()

    # genre_count
    genrewise_data = pickle.load(open("./pickle_files/genrewise_data.pkl", "rb"))

    # sample datapoints from bigger dataset
    sampled_data = random_sample(reqd_df, genrewise_data, top_k=10, size= 100)

    # preprocess the text: remove stopwords, lemmatize the text etc...s
    sampled_data["processed_textual_information"] = sampled_data["textual_information"].apply(lambda x: preprocess_tokens(porter_stemmer, x))

    
    sampled_data.to_csv(config["processed_data_path"], index=False)

    # initialize the tf-idf vectorizer
    tfidf = TfidfVectorizer(max_features=config["n_features"], min_df=2, ngram_range=(1,1))
    # fit the vectorizer
    vectorized_data = tfidf.fit_transform(sampled_data['processed_textual_information'].values)

    save_tfidf_vectorizer(tfidf, vectorized_data, config)
    
    return reqd_df, tfidf, vectorized_data


@validate
def run_similarity_search(config, sent):
    """Run the similarity search for the given user-query and return the top-5 matching movies."""
    
    # Check if the config has the data paths correct
    if not config["validity"]:
        raise Exception("There is no data path available")
    
    # if training is vlid, build the tokenizer
    if bool(config["train"]):
        reqd_df, vectorizer, vectorized_data = create_tfidf_embedding(config)

    # use pretrained tfidf vectorizer
    else:
        print("we already have the sampled data and the vectorizer")
        reqd_df = load_data(config)
        vectorizer, vectorized_data = load_tfidf_vectorizer(config)

    
    #lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()


    # preprocess user-query
    preprocessed_sent = preprocess_tokens(porter_stemmer, sent)
    # vectorize user-query
    vector = vectorizer.transform([preprocessed_sent])
    # compute cosine similarity
    sim_= cosine_similarity(vectorized_data, vector)
    # return top 5 similar movies
    score_dict = compute_top_5_titles(sim_, reqd_df)
    return score_dict


if __name__ == "__main__":    
    run_nltk_dependencies()
    result = run_similarity_search(config,example_sent)
    for title in result:
        item = result[title]
        print(f"Title: {title}, Score:{item[0]}")