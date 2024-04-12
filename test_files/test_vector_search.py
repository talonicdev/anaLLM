import ast
import os
import logging
import pytest

import numpy as np
import pandas as pd

from pathlib import Path
from decouple import config

from vector_search import VectorSearch


logging.basicConfig(filename='logs/test_vector.log',
                    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


os.environ['API_USER'] = config('USER')
os.environ['OPENAI_API_KEY'] = config('KEY')
os.environ['API_KEY'] = config('API_KEY')


class TestVectorDB:
    @pytest.fixture(autouse=True)
    def set_setter(self):
        cwd = Path(__file__).resolve().parent
        self.meta_data = pd.read_csv(f'{cwd}/test_data/test_meta_data_table.csv')
        self.vs = VectorSearch(debug=True)

    def test_embed_context(self):
        errors = []
        self.vs.embed_context(meta_data=self.meta_data)

        if not self.vs.me.collection.count() == 43:
            errors.append("Number of entries in Chroma DB does not match number of entries in meta data table.")

        test_table = self.vs.me.collection.get(
            ids=['2387eg_0',
                 '2387eg_1',
                 '2387eg_2',
                 '2387eg_3',
                 '2387eg_4',
                 '2387eg_5',
                 '2387eg_6',
                 '2387eg_7',
                 '2387eg_8',
                 '2387eg_9']
        )

        pred_doc = test_table['documents']
        cwd = Path(__file__).resolve().parents[0]
        f = open(f"{cwd}/test_data/example_document.txt", "r")
        x = f.read()
        true_doc = ast.literal_eval(x)

        if not len(pred_doc) == len(true_doc):
            errors.append("Number of entries in Documents does not match number of columns for this table.")

        if not np.all([1 if pred_doc[i] == true_doc[i] else 0 for i in range(len(true_doc))]):
            errors.append("Documents has wrong formatting.")

        if not np.all([list(d.keys())[0].isnumeric() for d in test_table['metadatas']]):
            errors.append("Chroma metadata keys are not integers.")

        assert not errors, "\n".join(errors)

    def test_find_tables(self):
        errors = []
        self.vs.embed_context()
        table_keys = self.vs.find_tables('How much pay student for netflix?')

        pred_prob = {self.vs.me.tables[i]: self.vs.me.table_probabilities[i] for i in range(len(self.vs.me.tables))}
        true_prob = {'2387eg': 0.5128275156021112,
                     'd38zbc': 0.1758841425180432,
                     'd38gqd4': 0.105317533016205}

        res = []
        for t in self.vs.me.tables:
            res.append(np.round(pred_prob[t], 2) == np.round(true_prob[t], 2))

        if not np.all(res):
            errors.append("Wrong probabilities calculated in vector search.")

        if not (table_keys[0] == '2387eg'):
            errors.append("Wrong VectorSearch results for 'How much pay student for Netflix'.")

        assert not errors, "\n".join(errors)
