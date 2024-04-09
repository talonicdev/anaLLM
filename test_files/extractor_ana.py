import ast
import os
import logging
import pytest
import uuid
import glob

import numpy as np
import pandas as pd

from pathlib import Path
from decouple import config

from complete import CompleteTable
from prebuilt import Extractor
from request_engine import TableSetter
from vector_search import VectorSearch


logging.basicConfig(filename='logs/test_request.log',
                    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


os.environ['API_USER'] = config('USER')
os.environ['OPENAI_API_KEY'] = config('KEY')
os.environ['API_KEY'] = config('API_KEY')


class TExtractor:
    # pytest --tb=line
    def __init__(self):
        '''files = glob.glob("/Users/stella/talonic/anaLLM2/anaLLM/test_files/test_data/quality_data/*.xlsx")

        for file in files:
            if Path(file).name != "quality_meta.xlsx":
                sheet = uuid.uuid4()
                self.setter = TableSetter(openai_api_key=os.environ['KEY'],
                                          api_key='',
                                          token='',
                                          sheet_id=str(sheet),
                                          test_file=Path(file).name,
                                          debug=True)

                self.setter.run(Path(file).name)
                self.setter.save_results()'''

        self.customer_request = "List all customers that purchased for over 250 Euro."

        '''self.extraction = Extractor(os.environ['KEY'],
                                    token='',
                                    api_key=os.environ['API_KEY'],
                                    customer_request=self.customer_request,
                                    debug=True,
                                    make_plot=True)

        self.extraction.get_meta_template()'''

        ct = CompleteTable(openai_api_key=os.environ['KEY'],
                           table_path='/Users/stella/talonic/anaLLM2/anaLLM/test_files/test_data/quality_data/out_table.xlsx',
                           token='',
                           users_prompt=self.customer_request,
                           api_key=os.environ['API_KEY'],
                           sheet_id='',
                           debug=True)
        e_cols, o_cols = ct.get_empty_cols()
        c_request, useful, exists_cols, empty_cols = ct.get_table_question(e_cols, o_cols)
        ct.create_table(c_request, useful, exists_cols, empty_cols)

    def test_select_tables(self):
        self.extraction.select_tables()

    def test_run_request(self):
        self.extraction.run_request()
        table = self.extraction.response
        cwd = Path(__file__).resolve().parent
        true_answer = pd.read_excel(f'{cwd}/test_data/purchase_table.xlsx', index_col=0)
        cols = [col for col in table.columns if 'ID' in col]
        count = 0
        for i, row in iter(table.iterrows()):
            if row[cols[0]] in true_answer['Customer ID']:
                count += 1

        # res = count / len(table)


if __name__ == '__main__':
    e = TExtractor()
    e.test_select_tables()
    e.test_run_request()

