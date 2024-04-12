import os
import ast
import logging
import chromadb
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from decouple import config
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from chromadb.config import Settings

from sentence_transformers import SentenceTransformer, util
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from utils.conf import load_templates, get_template, load_datasets
from common import Common, WriteType
from config import Config


os.environ['API_USER'] = config('USER')
os.environ['OPENAI_API_KEY'] = config('KEY')


logging.basicConfig(filename='logs/vector_search.log',
                    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


class MetaEngine:

    def __init__(self,
                 token: str = None,
                 debug: bool = False):

        self.token = token
        self.debug = debug
        self.config = Config()
        self.common = Common(config=self.config)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.corpus_embeddings = torch.zeros(size=(1, 0))

        self.tables = None
        self.collection = None
        self.chroma_client = None
        self.collection_dict = None
        self.table_probabilities = None

        self.init_client()

    def init_client(self):
        if not self.debug:
            proxy_host = "https://chroma-proxy.vhi.ai"
            proxy_port = 443
            try:
                self.chroma_client = chromadb.HttpClient(
                    host=proxy_host,
                    port=proxy_port,
                    settings=Settings(
                        chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                        chroma_client_auth_credentials=self.token
                    )
                )
            except Exception as e:
                self.common.write(WriteType.ERROR,f"Could not connect to '{self.token}'",None,e)
                raise SystemExit
        else:
            cwd = Path(__file__).resolve().parents[0]
            self.chroma_client = chromadb.PersistentClient(path=f"{cwd}/test_files/chromadb")

    def create_new_collection(self, collection_name):
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def load_collection(self, collection_name):
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def embed_new_vec(self, vec_num, sheet_id, corpus):
        new_embed = self.embedder.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
        if self.corpus_embeddings.nelement() == 0:
            self.corpus_embeddings = new_embed
        else:
            self.corpus_embeddings = torch.cat((self.corpus_embeddings, new_embed))

        self.save_vec(vec_num, sheet_id, new_embed.tolist(), corpus)

    def save_vec(self, vec_num, sheet_id, embeddings, corpus):
        # collection.upsert(...)

        self.collection.add(
            embeddings=embeddings,
            documents=corpus,
            metadatas=[{f'{vec_num + i}': f'{[sheet_id, c]}', 'sheet_id': sheet_id} for i, c in enumerate(corpus)],
            ids=[f'{sheet_id}_{i}' for i in range(len(corpus))]
        )

    def load_vec(self):
        self.collection_dict = self.collection.get()
        dataset = self.collection.get(include=['embeddings'])
        self.corpus_embeddings = torch.FloatTensor(dataset['embeddings'])

    def find_semantic(self, queries: List[str], n_results=1000):
        column_dict = {int(k): v for d in self.collection_dict['metadatas'] for k, v in d.items()}
        self.tables = list(set([ast.literal_eval(column_dict[d])[0] for d in column_dict.keys()]))
        self.table_probabilities = np.zeros(len(self.tables))

        for query in queries:
            self.common.write(WriteType.DEBUG, f'Performing semantic search for "{query}"..')
            table_probability = {t: [] for t in self.tables}

            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=n_results)
            hits = hits[0]

            for idx, hit in enumerate(hits):
                table_probability[ast.literal_eval(column_dict[hit['corpus_id']])[0]].append(hit['score'])

            table_scores_list = list(table_probability.values())
            table_scores = np.zeros([len(table_scores_list), len(max(table_scores_list, key=lambda x: len(x)))])
            for i, j in enumerate(table_scores_list):
                table_scores[i][0:len(j)] = j

            # options: table_max_scores = np.mean(table_scores, axis=1) or table_max_scores = np.max(table_scores, axis=1)

            table_max_scores = np.mean(table_scores, axis=1)
            self.table_probabilities += table_max_scores / len(queries)

        indices = [list(self.table_probabilities).index(p) for p in self.table_probabilities if p > 0.3]
        return [self.tables[idx] for idx in indices]


class VectorSearch:
    def __init__(self,
                 token: str = None,
                 debug: bool = False,
                 reload: bool = False,
                 collection_name: str = 'talonic_collection'):

        self.debug = debug
        self.reload = reload
        self.collection_name = collection_name if not debug else 'test_collection'
        self.me = MetaEngine(token=token, debug=True)
        self.initialize()
        self.me.load_collection(self.collection_name)

    def initialize(self):
        if self.reload:
            self.me.load_collection('test_collection')
            if self.me.collection.count() > 0:
                self.me.chroma_client.delete_collection(name="test_collection")

    def embed_context(self, vec_num: int = 0, meta_data: pd.DataFrame = None):
        for idx, row in meta_data.iterrows():
            keywords = ast.literal_eval(row['keywords'])
            all_keywords = keywords + [row['table_name']]
            k = ' - '.join(all_keywords)
            column_names = ast.literal_eval(row['column_names'])
            concat = [' - '.join([k, c]) for c in column_names]
            self.me.embed_new_vec(vec_num, sheet_id=row['key'], corpus=concat)
            vec_num += len(concat)

    def find_tables(self, request: str):
        self.me.load_vec()
        table_keys = self.me.find_semantic([request])
        return table_keys
