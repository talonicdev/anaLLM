import ast
import os
import logging
import pytest

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


class TestTableSetter:
    @pytest.fixture(autouse=True)
    def set_setter(self):

        sheet_id = 'd38zbc'
        self.setter = TableSetter(openai_api_key=os.environ['KEY'],
                                  api_key='',
                                  token='',
                                  sheet_id=sheet_id,
                                  test_file='test_meta_data_table.csv',
                                  debug=True)

        cwd = Path(__file__).resolve().parents[1]
        self.setter.table = pd.read_csv(
            f'{cwd}/test_files/test_data/Customer Reports - Customer Reports.csv')

        self.setter.table['avg_income'] = [str(np.round(n, 2)) + ' euro' for n in
                                           np.random.uniform(10, 1000000, len(self.setter.table))]
        self.setter.table.rename(columns={'Income_Category': 'unnamed(0)'}, inplace=True)

        self.setter.dataset_name = 'Customer Reports'
        self.setter.key = sheet_id
        self.setter.setup_entry()

    def test_get_column_description(self):
        column_descriptions = self.setter.get_column_description()
        new_cols = ast.literal_eval(column_descriptions)
        new_cols = [c.lower() for c in new_cols]
        old_cols = [c.replace('_', ' ') for c in self.setter.table.columns.tolist()]
        old_cols = [c.lower() for c in old_cols]
        assert f'{new_cols[1:]}' == f'{old_cols[1:]}'

    def test_columns_info(self):
        errors = []
        self.setter.columns_info()

        if not len(ast.literal_eval(self.setter.meta_data_table['column_names'][0])) == self.setter.table.columns.size:
            errors.append("Number of column_names is unequal to number of generated column_names.")

        if not isinstance(
            ast.literal_eval(self.setter.meta_data_table['column_names'][0]), list
        ):
            errors.append("Type of column_names is not list.")

        assert not errors, "\n".join(errors)

    def test_get_summary(self):
        errors = []
        self.setter.get_summary()
        answer = self.setter.description

        if not len(answer) > 0:
            errors.append("Summary answer is empty.")

        if not len(answer.split('"')) == 1:
            errors.append("Wrong formatting of Summary answer.")

        if not isinstance(
            ast.literal_eval(self.setter.meta_data_table['keywords'][0]), list
        ):
            errors.append("Type of column_names is not list.")

        if not len(ast.literal_eval(self.setter.meta_data_table['keywords'][0])) > 0:
            errors.append("No keyword saved.")

        assert not errors, "\n".join(errors)

    def test_get_column_types(self):
        errors = []
        cwd = Path(__file__).resolve().parents[0]
        self.setter.get_column_types()
        pred_units = pd.DataFrame(self.setter.table[['avg_income', 'unit']])
        true_units = pd.read_csv(f'{cwd}/test_data/units.csv')

        false_numerics_vals = [a for a, b in zip(pred_units, true_units) if a != b]
        c_types = self.setter.col_types

        if not len(false_numerics_vals) == 0:
            errors.append("Numeric values are wrong formatted.")

        if not c_types.avg_income == 'float64':
            errors.append("Wrong data type for column 'avg_income'.")

        if not isinstance(
                ast.literal_eval(self.setter.meta_data_table['column_types'][0]), list
        ):
            errors.append("Type of column_types is not list.")

        assert not errors, "\n".join(errors)

    def test_context(self):
        errors = []
        self.setter.context()
        if not len(self.setter.context_result) > 0:
            errors.append("Context answer is empty.")

        if not isinstance(self.setter.context_result, str):
            errors.append("Context answer is not string type.")

        assert not errors, "\n".join(errors)
