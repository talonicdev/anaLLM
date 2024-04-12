import ast
import os
import logging
import pytest
import uuid
import csv

import numpy as np
import pandas as pd

from pathlib import Path
from decouple import config

from request_engine import TableSetter
from vector_search import VectorSearch


logging.basicConfig(filename='logs/quality_analysis.log',
                    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)

os.environ['API_USER'] = config('USER')
os.environ['OPENAI_API_KEY'] = config('KEY')
os.environ['API_KEY'] = config('API_KEY')


class QualityAnalysis:

    def __init__(self):
        pass

    def set_tables(self):
        cwd = Path(__file__).resolve().parent
        test_files = [f.name for f in list(Path(f"{cwd}/test_data/quality_data").glob("*.xlsx"))]

        for test_file in test_files:
            setter = TableSetter(openai_api_key=os.environ['KEY'],
                                 api_key='',
                                 token='',
                                 sheet_id=str(uuid.uuid4()),
                                 test_file=test_file,
                                 debug=True)
            setter.run()

    def quality_vector(self):
        cwd = Path(__file__).resolve().parent
        meta_data = pd.read_csv(f'{cwd}/test_data/test_meta_data_table.csv')
        v = VectorSearch(meta_data=meta_data, debug=True)
        v.embed_context()
        table_keys = v.find_tables('How much pay students for netflix?')
        print(table_keys)


# quality_vector()
QualityAnalysis().set_tables()
