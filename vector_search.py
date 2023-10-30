import re
import os

import yaml

import numpy as np
import pandas as pd

from numpy.linalg import norm
from pathlib import Path

import chromadb
# import tensorflow as tf
# import tensorflow_hub as hub

from sentence_transformers import SentenceTransformer, util
import torch


class USE_engine:
    columns1 = ['sex', 'Retailer', 'Retailer ID', 'Invoice Date', 'Region', 'State', 'City', 'Product',
                'Price per Unit', 'Units Sold', 'Total Sales', 'Operating Profit', 'Operating Margin', 'Sales Method']
    columns2 = ['User ID', 'Subscription Type', 'Monthly Revenue', 'Join Date', 'Last Payment Date', 'Country', 'Age',
                'Gender', 'Device', 'Plan Duration']
    columns3 = ['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)', 'Sales ($)', 'gender']

    messages = columns1 + columns2 + columns3

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))
    def embed(self, msgs):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        # model = hub.load(module_url)
        # return model(msgs)

    def create_USE_embeds(self, messages):
        message_embeddings = self.embed(messages)

        message_embeddings_dict = {i: m.tolist() for i, m in enumerate(message_embeddings.numpy())}
        with open('./message_embeddings.yaml', 'w', ) as f:
            yaml.dump(message_embeddings_dict, f, sort_keys=False)

        print('messages encoded!')

    def find_results(self):
        if Path('message_embeddings.yaml').is_file():
            with open('message_embeddings.yaml', 'r') as f:
                message_embeddings = yaml.safe_load(f)

        message_embeddings = [np.array(message_embeddings[m]) for m in message_embeddings.keys()]

        query_embedding = self.embed(["gender"])

        print("Query: gender")
        for e, s in zip(message_embeddings, self.messages):
            print(s, " -> similarity score = ",
                 self.cosine_similarity(e, query_embedding))


class MetaEngine:

    def __init__(self, collection_name: str):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        chroma_client = chromadb.Client()
        self.collection = chroma_client.get_or_create_collection(name=collection_name)

        self.corpus_embeddings = torch.zeros(size=(1, 0))
        self.collection_dict = None

    def embed_new_vec(self, idx, corpus):
        new_embed = self.embedder.encode(corpus, convert_to_tensor=True)
        if self.corpus_embeddings.nelement() == 0:
            self.corpus_embeddings = new_embed
        else:
            self.corpus_embeddings = torch.cat((self.corpus_embeddings, new_embed))

        self.save_vec(idx, new_embed.tolist(), corpus)

    def save_vec(self, idx, embeddings, corpus):
        self.collection.add(
            embeddings=embeddings,
            documents=corpus,
            metadatas=[{f'{idx}_{i}': c} for i, c in enumerate(corpus)],
            ids=[f'{idx}_{i}' for i in range(len(corpus))]
        )

    def load_vec(self):
        self.collection = self.collection.get()
        dataset = self.collection.get(include=['embeddings'])
        self.corpus_embeddings = dataset['embeddings']

        meta = self.collection_dict['metadatas']
        ids = self.collection_dict['ids']

    def find_cos(self, corpus, query: str, n_results=5, threshold=0.3):
        top_k = n_results
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)

        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", query)
        print(f"\nTop {n_results} most similar sentences in corpus:\n")

        for score, idx in zip(top_results[0], top_results[1]):
            if score >= threshold:
                print(corpus[idx], "(Score: {:.4f})".format(score))

    def find_sem(self, corpus, query: str, n_results=5, threshold=0.3):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=n_results)
        hits = hits[0]

        print("\n\n======================\n\n")
        print("Query:", query)
        print(f"\nTop {n_results} most similar sentences in corpus:\n")
        for hit in hits:
            if hit['score'] >= threshold:
                print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))


if __name__ == "__main__":
    columns1 = ['sex', 'Retailer', 'Retailer ID', 'Invoice Date', 'Region', 'State', 'City', 'Product',
                'Price per Unit', 'Units Sold', 'Total Sales', 'Operating Profit', 'Operating Margin', 'Sales Method']
    columns2 = ['User ID', 'Subscription Type', 'Monthly Revenue', 'Join Date', 'Last Payment Date', 'Country', 'Age',
                'Gender', 'Device', 'Plan Duration']
    columns3 = ['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)', 'Sales ($)', 'gender']

    query = 'females and males'

    me = MetaEngine('test_collection')

    corpus = columns1 + columns2 + columns3

    for idx, col in enumerate([columns1, columns2, columns3]):
        me.embed_new_vec(idx, col)

    me.load_vec()

    me.find_cos(corpus, query)
    me.find_sem(corpus, query)
