import nltk
import pandas as pd
import pickle
import numpy as np
import contractions
import os
import warnings
import random

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from functools import wraps
warnings.filterwarnings("ignore")



def random_sample(reqd_df, dk, top_k=10, size= 100):
    """Randomly sample data pertaining to every genre."""
    
    # The main aim of this function is to reduce the dataset size.
    # The original size of 18k and we are reducing it to 1k.
    # This sort of eases the computation
    final_dataset_indices = []
    visited_set = set()

    for index, (k,v) in enumerate(dk.items()):
                
        if index > 9:
            break
        
        flag = 0
        while not flag:

            prev_ = len(visited_set)
            indices = random.sample(v, size)
            visited_set.update(set(indices))
            new_  = len(visited_set)
            diff = new_ - prev_

            if (diff - len(indices)) <= 2:
                flag = 1
                final_dataset_indices.extend(indices)

    final_df = reqd_df.iloc[final_dataset_indices]
    return final_df
         



def validate(func):
    """Validate the config JSON file"""
    
    @wraps(func)
    def wrapper(config, *args, **kwargs):
        path = config.get("data_path")
        if path is None or path == "":
            config["validity"] = False
        
        elif not os.path.exists(path):  
            config["validity"] = False
        
        else:
            config["validity"] = True

        if len(os.listdir(config["vectorizer_save_path"])) > 1:
            config["train"] = 0

        return func(config, *args, **kwargs)  
    
    return wrapper


def run_nltk_dependencies():

    """Download and Install the NLTK dependencies"""
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    return



def load_data(config):
    """Load the Dataset"""
    # If we want to build the tokenizer from scratch
    # load the whole dataset
    if config["train"]:
        df = pd.read_csv(config["data_path"])
        return df

    # if we already have the processed data path
    elif os.path.exists(config["processed_data_path"]):
        print("dataset_present", "processed_path")
        df = pd.read_csv(config["processed_data_path"])
        return df
    # else raise exception
    else:
        raise Exception("Cannot Load Dataset")
        
def process_dataframe(df, config):
    
    """Get required rows and columns from the dataset."""

    # get the reqd columns for the task at hand.
    reqd_df = df[["imdb_id", "title", "plot_synopsis", "tags"]]
    reqd_df["tags"] = reqd_df["tags"].apply(lambda x: x.replace(",", " "))
    reqd_df["textual_information"] = reqd_df["tags"] + " " + reqd_df["plot_synopsis"]
    
    return reqd_df


def preprocess_tokens(lemmatizer, sent):
    """Perform Lemmatizations and contractions on the detaset. Remove stopwords etc.."""
    stop_words = dict.fromkeys(list(set(stopwords.words('english'))), 1)


    word_tokens = word_tokenize(sent)
    filtered_sentence = [contractions.fix(w.lower()) for w in word_tokens if stop_words.get(w) is None and w.isnumeric() is False]
    filtered_sentence_lemma = [lemmatizer.stem(w) for w in filtered_sentence]
    
    return " ".join(filtered_sentence_lemma)


def save_tfidf_vectorizer(vectorizer, vectorized_data, config):
    """Save the TF-IDF Vectorizer."""
    
    pickle.dump(vectorizer, open(os.path.join(config["vectorizer_save_path"], r"vectorizer.pkl"), "wb"))
    pickle.dump(vectorized_data, open(os.path.join(config["vectorizer_save_path"], r"vectorized_data.pkl"), "wb"))


def load_tfidf_vectorizer(config):
    """Load the TF-IDF Vectorizer"""

    vectorizer  = pickle.load(open(os.path.join(config["vectorizer_save_path"], r"vectorizer.pkl"), "rb"))
    vectorized_data = pickle.load(open(os.path.join(config["vectorizer_save_path"], r"vectorized_data.pkl"), "rb"))

    return vectorizer, vectorized_data


def compute_top_5_titles(sim_, df):

    """Get the top 5 recommendations and their scores."""
    score_dict = {}
    sorted_indices = np.argsort([x[0] for x in sim_])[::-1]
    top_5 = sorted_indices[:5]

    for ind in top_5:
        score_dict[df.iloc[ind]["title"]] = (sim_[ind][0], df.iloc[ind]["tags"])
    
    return score_dict
    

