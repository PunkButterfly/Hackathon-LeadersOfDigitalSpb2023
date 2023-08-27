from numpy import dot
from numpy.linalg import norm
import pandas as pd
from pydantic import BaseModel
import string
import re
import numpy as np
import torch

from .models import *

from fastDamerauLevenshtein import damerauLevenshtein

def embed_bert(default_bert, text):

    with torch.no_grad():
        model_output = default_bert(text)

    return model_output.cpu().numpy()

class Query(BaseModel):
    objects: list = []

def load_dataset(dataset_name):
    df = pd.read_csv(dataset_name)
    df = df[df['is_actual'] == True]
    return df

def replace_punctuation_with_spaces(text):
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(translator)

def add_spaces_around_digits(input_string):
    result = re.sub(r'(\d+)', r' \1 ', str(input_string))
    return result

def replace_by_list(token):
    vocab = {'дом':'д',
             'переулок':'п',
             'пер':'п',
             'улица':'у',
             'ул':'у',
             'поселок':'п',
             'пос':'п',
             'область':'о',
             'обл':'о',
             'строение':'с',
             'стр':'с',
             'город':'г',
             'гор':'г',
             'набережная':'н',
             'наб':'н',}
    if token in vocab:
      return vocab[token]
    return token

def convert_string(s, uniq_words = None):
    s = add_spaces_around_digits(s)
    s = replace_punctuation_with_spaces(s.lower()).split()
    s = [replace_by_list(x) for x in s]
    if uniq_words is None:
        return [x for x in s if x!='']
    else:
        return [x for x in s if x!='' and x in uniq_words]

def proceed_data(dataset_path, target_col = 'full_address'):
    data = load_dataset(dataset_path)
    data['proceed_target'] = data[target_col].apply(lambda x: convert_string(x))
    return data

def get_uAddresses(dataset_path):
    dataset = load_dataset(dataset_path)
    return list(set(list(dataset['target_address'])))

def get_uWords(uniq_addres):
    words = set()
    for i in uniq_addres:
        words.update(i)
    return list(words)

def get_lev_embeed(query_text, lev_word_tokenizer, uniq_words):
    return words2nums(lev_word_tokenizer, convert_string(query_text, uniq_words))

def words2nums(word2num, words):
  return [word2num[x] for x in words]

def nums2words(num2word, nums):
  return [num2word[x] for x in nums]

def cosine(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def get_lev_metric(v1, v2):
  return damerauLevenshtein(v1, v2, similarity=False, deleteWeight=50, insertWeight=1, replaceWeight=50, swapWeight=20)

def get_elems_for_ml_ranking(DB, top_k=10):
    return DB.sort_values('lev_dist_to_key').head(top_k)

def sort_by_similarity(query_text, default_bert=None, trained_bert=None, DB=None, lev_word_tokenizer=None,uniq_words=None):
    
    
    lev_embeed = get_lev_embeed(query_text, lev_word_tokenizer, uniq_words)
    DB['lev_dist_to_key'] = DB['tokenize_addresses'].apply(lambda x: get_lev_metric(lev_embeed, x))

    top_base = get_elems_for_ml_ranking(DB, top_k=10)

    # return top_base

    key_bert_embeed = embed_bert(trained_bert, query_text)
    top_base['target_embeed'] = top_base['full_address'].progress_apply(lambda x: embed_bert(default_bert, x))
    top_base['bert_dist_to_key'] = top_base['target_embeed'].apply(lambda x : cosine(x[0], key_bert_embeed[0]))

    top_base = top_base.sort_values('bert_dist_to_key', ascending=False)
    
    # return top_base.sort_values('bert_dist_to_key', ascending=False)
    # ДОБАВИЛ ПЕРЕРАНЖИРОВАНИЕ 

    equals_first = top_base[top_base['lev_dist_to_key'] == top_base['lev_dist_to_key'].iloc[0]]
    top_base.iloc[0:equals_first.shape[0], :] = equals_first.sort_values('full_address')

    return top_base



def search_object_in_base(query_text, globals):

    top_k_res = sort_by_similarity(query_text, **globals)
    
    return top_k_res

def make_response(queries, globals):
    responses = []
    for query_text in queries:
        top_k_res = search_object_in_base(query_text, globals)
        responses.append(
            {
            "target_building_id": el['id'],
            "target_address": el['full_address'],
            # "bert_dist_to_key": el['bert_dist_to_key'],
            # "lev_dist_to_key": el['lev_dist_to_key'],
            }
         for idx, el in top_k_res.iterrows()
        )
    return True, responses