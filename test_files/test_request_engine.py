import ast
import numpy as np
import os
import pandas as pd
import pytest

from pathlib import Path
from decouple import config

from complete import CompleteTable
from prebuilt import Extractor
from request_engine import TableSetter
from utils.conf import parse_string, parse_number, is_date, is_range


os.environ['API_USER'] = config('USER')
os.environ['OPENAI_API_KEY'] = config('KEY')
os.environ['API_KEY'] = config('API_KEY')


def test_set_table():
    setter = TableSetter(openai_api_key=os.environ['KEY'],
                         api_key='',
                         token='',
                         sheet_id='',
                         new_collection=False,
                         debug=True)
    cwd = Path(__file__).resolve().parents[1]
    setter.table = pd.read_csv(
        f'{cwd}/test_files/Customer Reports - Customer Reports.csv')

    setter.table['avg_income'] = [str(np.round(n, 2)) + ' euro' for n in np.random.uniform(10, 1000000, len(setter.table))]
    setter.table.rename(columns={'Income_Category': 'unnamed(0)'}, inplace=True)
    setter.dataset_name = 'Customer Reports'
    setter.key = '129d8h'
    column_descriptions = setter.get_column_description()
    new_cols = ast.literal_eval(column_descriptions)
    new_cols = [c.lower() for c in new_cols]
    old_cols = [c.replace('_', ' ') for c in setter.table.columns.tolist()]
    old_cols = [c.lower() for c in old_cols]
    assert f'{new_cols[1:]}' == f'{old_cols[1:]}'


def test_always_passes():
    assert True


def test_always_fails():
    assert False
