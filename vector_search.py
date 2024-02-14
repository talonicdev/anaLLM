import os
import ast
import logging
from pathlib import Path
from typing import List

import copy

import numpy as np
import pandas as pd
import torch
import yaml

from decouple import config
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer, util
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from utils.conf import load_templates, get_template, load_datasets


os.environ['API_USER'] = config('USER')
os.environ['OPENAI_API_KEY'] = config('KEY')


logging.basicConfig(filename='prebuilt.log', format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)


class BaseEmbedding:

    def __init__(self,
                 token: str):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        proxy_host = "https://chroma-proxy.vhi.ai"
        proxy_port = 443

        # Initialize the chroma client as an HTTP client
        self.chroma_client = chromadb.HttpClient(
            host=proxy_host,
            port=proxy_port,
            settings=Settings(
                chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                chroma_client_auth_credentials=token
            )
        )

        self.corpus_embeddings = torch.zeros(size=(1, 0))
        self.collection_dict = None
        self.collection = None

    def create_new_collection(self, collection_name):
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def load_collection(self, collection_name):
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def embed_new_vec(self, vec_num, table_index, corpus):
        new_embed = self.embedder.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
        if self.corpus_embeddings.nelement() == 0:
            self.corpus_embeddings = new_embed
        else:
            self.corpus_embeddings = torch.cat((self.corpus_embeddings, new_embed))

        self.save_vec(vec_num, table_index, new_embed.tolist(), corpus)

    def save_vec(self, vec_num, table_index, embeddings, corpus):
        self.collection.add(
            embeddings=embeddings,
            documents=corpus,
            metadatas=[{f'{vec_num + i}': f'{(table_index, c)}'} for i, c in enumerate(corpus)],
            ids=[f'{vec_num + i}' for i in range(len(corpus))]
        )

    def load_vec(self):
        self.collection_dict = self.collection.get()
        dataset = self.collection.get(include=['embeddings'])
        self.corpus_embeddings = torch.FloatTensor(dataset['embeddings'])


class MetaEngine(BaseEmbedding):
    def __init__(self, token: str):
        super().__init__(token)

    def find_semantic(self, query: str, n_results=100, threshold=0.3):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=n_results)
        hits = hits[0]

        column_dict = {ast.literal_eval(list(d.keys())[0]): ast.literal_eval(list(d.values())[0]) for d in self.collection_dict['metadatas']}
        similar_results = []
        logging.info(f"Similarity Result:")
        for hit in hits:
            if hit['score'] >= threshold:
                similar_results.append(column_dict[hit['corpus_id']])
                logging.info(f"Element: {column_dict[hit['corpus_id']]} - Score: {hit['score']}")

        return similar_results

    def show_find_cos(self, corpus, query: str, n_results=5, threshold=0.3):
        top_k = n_results
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)

        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("COSINE")
        print("Query:", query)
        print(f"\nTop {n_results} most similar sentences in corpus:\n")

        for score, idx in zip(top_results[0], top_results[1]):
            if score >= threshold:
                print(corpus[idx], "(Score: {:.4f})".format(score))

    def show_find_sem(self, corpus, query: str, n_results=5, threshold=0.3):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=n_results)
        hits = hits[0]

        print("\n\n======================\n\n")
        print("SEMANTIC")
        print("Query:", query)
        print(f"\nTop {n_results} most similar sentences in corpus:\n")
        for hit in hits:
            if hit['score'] >= threshold:
                print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))


class ContextCluster(BaseEmbedding):

    def __init__(self, collection_name: str):
        super().__init__(collection_name)

        self.score = -1
        self.scores_list = []
        self.group_index = None
        self.prompt_template = None

        self.template = None
        self.prefix = None
        self.suffix = None
        self.examples = None

        self.columns = None
        self.original_col = None
        self.original_embeds = None

        self.tables: List[pd.DataFrame] = []

        self.batch_size = 3
        self.max_iter = 5

        self.params = {}

    def get_clusters(self, corpus: List[str]):
        groups = {}

        for i in range(len(corpus)):
            if self.group_index[i] in list(groups.keys()):
                groups[self.group_index[i]].append(corpus[i])
            else:
                groups[self.group_index[i]] = [corpus[i]]

        print(groups)
        return groups

    def kmeans_clusters(self, n_cluster=7):
        kmeans = MiniBatchKMeans(n_clusters=n_cluster,
                                 reassignment_ratio=0,
                                 random_state=0,
                                 batch_size=self.batch_size,
                                 max_iter=self.max_iter,
                                 n_init="auto").fit(self.corpus_embeddings)

        self.group_index = kmeans.fit_predict(self.corpus_embeddings)

    def agglomerative_clusters(self):
        clustering = AgglomerativeClustering().fit(self.corpus_embeddings)
        self.group_index = clustering.labels_

    def silhouette_score(self):
        """
        best value = 1
        worst value = -1
        :return:
        """
        self.score = silhouette_score(self.corpus_embeddings, self.group_index)
        print("\nSCORE: ")
        print(self.score)

    def get_meta_template(self):
        self.template, self.prefix, self.suffix, self.examples = load_templates('cluster_template')
        self.prompt_template = get_template(self.examples,
                                            self.prefix,
                                            self.suffix)

    def predict_topic(self, columns):
        prompt = self.prompt_template.format(data=columns)
        prompt_template = ChatPromptTemplate.from_template(prompt)
        message = prompt_template.format_messages()

        llm = ChatOpenAI(temperature=0, openai_api_key=config('KEY'))
        response = llm(message)

        return response

    def find_centers(self):
        # find best cluster representations
        pass

    def init_data(self):
        self.tables = load_datasets(subset=3)
        self.columns = [df.columns.values.tolist() for df in self.tables]

        vec_num = 0

        for col, table_index in zip(self.columns, [i for i in range(len(self.columns))]):
            self.embed_new_vec(vec_num, table_index, col)
            vec_num += len(col)

        self.columns = [col for col_df in self.columns for col in col_df]
        self.original_col = copy.deepcopy(self.columns)
        self.original_embeds = copy.deepcopy(self.corpus_embeddings)

    def find_best_topic(self, round=1, n_clusters=3):
        print(f"\n\n\n========= ROUND: {round} =========")
        print(f"\nNumber Cluster: {n_clusters}")
        print(f"Max Iter: {self.max_iter}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Calculations: {self.batch_size * self.max_iter}\n")

        self.kmeans_clusters(n_clusters)
        groups = self.get_clusters(self.columns)
        self.silhouette_score()
        self.scores_list.append(self.score)

        if (self.score > 0.5) or (n_clusters == 10):
            print(groups)
            return

        n_clusters += 1

        for num in groups.keys():
            print("\n---")
            if len(groups[num]) > 2:
                topic = self.predict_topic(groups[num])
                print(f"column size : {len(self.columns)}")
                if topic.content not in self.columns:
                    print(f"\ndata size: {len(groups[num])}")
                    print(topic.content)
                    self.columns += [topic.content]
                    print("Embedding new topic ...")
                    topic_embedding = self.embedder.encode(topic.content, convert_to_tensor=True)
                    topic_embedding = topic_embedding.expand(1, topic_embedding.shape[0])
                    self.corpus_embeddings = torch.cat((self.corpus_embeddings, topic_embedding))
                    print("Done")
            else:
                print(f"\nSkip cluster number: {num} of total {n_clusters}")

        self.find_best_topic(round, n_clusters)

    def run(self):
        self.get_meta_template()
        self.init_data()

        t = np.linspace(7, 15, 15)

        batch = lambda x: x * (np.cos(np.pi/4) - np.sin(4 * x) * np.sin(np.pi/4))
        iter = lambda x: x * (np.sin(np.pi/4) + np.sin(4 * x) + np.cos(np.pi/4))

        inter_vals = iter(t)
        batch_vals = batch(t)

        count = 0

        for i, b in zip(inter_vals, batch_vals):

            count += 1

            if np.floor(b) <= 10 and np.floor(i) <= 25:
                self.batch_size = int(np.floor(b)) if int(np.floor(b)) > 3 else 3
                self.max_iter = int(np.floor(i)) if int(np.floor(i)) > 5 else 5

                self.scores_list = []
                self.columns = copy.deepcopy(self.original_col)
                self.corpus_embeddings = copy.deepcopy(self.original_embeds)
                self.find_best_topic(round=count)

                best_score = np.max(np.array(self.scores_list))
                n_clusters = self.scores_list.index(best_score) + 3

                self.params[best_score] = {'batch': self.batch_size, 'iter': self.max_iter, 'n_cluster': n_clusters}

        with open('./parameters.yaml', 'w', ) as f:
            yaml.dump(self.params, f, sort_keys=False)


if __name__ == "__main__":

    '''cluster = ContextCluster('test_collection')
    cluster.run()'''

    columns1 = ['Retailer', 'Retailer ID', 'Invoice Date', 'Region', 'State', 'City', 'Product', 'sex',
                'Price per Unit', 'Units Sold', 'Total Sales', 'Operating Profit', 'Operating Margin', 'Sales Method']
    columns2 = ['User ID', 'Subscription Type', 'Monthly Revenue', 'Join Date', 'Last Payment Date', 'Country', 'Age',
                'Gender', 'Device', 'Plan Duration']
    columns3 = ['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)', 'Sales ($)', 'gender']
    corpus = columns1 + columns2 + columns3

    query = 'females'
    me = MetaEngine('test_collection')
    vec_num = 0

    for col, table_index in zip([columns1, columns2, columns3], [1111, 2222, 3333]):
        me.embed_new_vec(vec_num, table_index, col)
        vec_num += len(col)

    me.load_vec()
    similar_results = me.find_semantic(query)
    print(similar_results)
    # me.show_find_cos(corpus, query)
    # me.show_find_sem(corpus, query)
