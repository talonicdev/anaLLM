import argparse
import ast
import json
import os
import logging
import requests

from aenum import extend_enum
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import openai
import re
import dateutil.parser as dparser
import yaml

from pathlib import Path

from pandasai.llm.openai import OpenAI
from langchain_openai import ChatOpenAI

from utils.conf import load_templates, get_template, WordContext, WordException, parse_string, parse_number, is_date, \
    is_range, format_dataframe, special_char, Units, check_number_unit
from vector_search import VectorSearch
from config import Config
from common import Common, WriteType, Requests

logging.basicConfig(filename='logs/request_engine.log',
                    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


class TableSetter:

    def __init__(self,
                 openai_api_key: str,
                 api_key: str,
                 token: str,
                 sheet_id: str,
                 format: str = 'excel',
                 test_file=None,
                 debug=False):

        """
        Creates a metadata table.
        :param api_key: key of openai key
        :param dataset_name: name of dataset with file extension, ex.: table.csv
        """

        self.api_key = api_key
        self.token = token
        self.format = format
        self.sheet_id = sheet_id
        self.openai_api_key = openai_api_key
        self.debug = debug

        self.config = None
        self.common = None
        self.requests = None
        self.meta_data_table = None
        self.llm = None
        self.tables = None
        self.vs = None

        self.key = None
        self.table = None
        self.dataset_name = None
        self.destination_name = None
        self.column_description = None
        self.table_side_data = None
        self.column_names = None
        self.user_id = None

        self.description = None
        self.keywords = None
        self.col_types = None
        self.context_result = None

        self.template = None
        self.examples = None
        self.prefix = None
        self.suffix = None

        self.prompt_template = None
        self.terminate = False

        self.init_attr(openai_api_key, token, api_key, debug=debug, test_file=test_file)
        self.load_WordContext()
        self.load_Units()

    def init_attr(self, openai_api_key, token, api_key, debug=False, test_file=None):
        if not debug:
            self.config = Config(
                openai_api_key=openai_api_key,
                token=token,
                api_key=api_key
            )
            self.common = Common(config=self.config)
            self.requests = Requests(config=self.config)
            self.load_table()

        else:
            cwd = Path(__file__).resolve().parent
            self.table, self.table_side_data = format_dataframe(f'{cwd}/test_files/test_data/quality_data/{test_file}')

        os.environ['OPENAI_API_KEY'] = openai_api_key

        columns = ['key', 'table_name', 'creation_date', 'last_update', 'context', 'keywords',
                   'column_names', 'description', 'column_types']
        if self.debug: columns.append('path')
        self.meta_data_table = pd.DataFrame(columns=columns)

        self.llm = OpenAI(api_token=openai_api_key, engine="gpt-4-1106-preview")

        name = 'test_collection' if self.debug else self.config.COLLECTION_NAME
        self.vs = VectorSearch(self.token, debug=self.debug, collection_name=name)

    def load_table(self):
        base_url = 'https://backend.vhi.ai/service-api'
        headers = {'Authorization': f'Bearer {self.token}',
                   'x-api-key': f'{self.api_key}'}
        response = requests.get(f"{base_url}/sheet/{self.sheet_id}", headers=headers)
        if response.status_code == 200:
            sheet_data = response.json()
            self.table_side_data = f"{sheet_data['tableName']} - {sheet_data['sheetName']}"

            if self.format == 'excel':
                self.table, self.table_side_data = format_dataframe(sheet_data['sheet'])
            else:
                self.table = pd.DataFrame(sheet_data['sheet'])

            self.user_id = sheet_data['userId']
        else:
            print("Error:", response.status_code, response.text)

    def update_table(self):
        """
        If entry in metadata table already exists and just has some changes.
        :return: Updated metadata table for given key of changed table.
        """
        date_time = datetime.fromtimestamp(datetime.timestamp(datetime.now()))
        last_update = date_time.strftime("%d-%m-%Y, %H:%M:%S")
        self.table.loc[self.table['key'] == self.key, 'last_update'] = last_update

    def process_entries(self):
        """
        Inserts values for all columns of row entry with given key.
        """
        self.columns_info()
        self.get_summary()
        self.get_column_types()
        self.context()

    def save_results(self):
        self.load_into_vector_db()
        self.save_meta_data_table()

    def run(self, test_file=None):
        """
        Updates or initialises entry in metadata table.
        :param destination_name: new short name for given table
        :param table_key: if entry already exists this key is a reference to given table.
        """
        self.setup_entry(test_file=test_file)
        self.process_entries()

    def setup_entry(self, test_file=None) -> None:
        """
        Creates new entry in meta_data_table w/ key, table_name, creation_date, last_update
        """
        key = datetime.timestamp(datetime.now())
        table_name = ' '.join(self.table_side_data)
        date_time = datetime.fromtimestamp(key)
        last_update = creation_date = date_time.strftime("%d-%m-%Y, %H:%M:%S")
        self.key = self.sheet_id

        empty = 6 if self.debug else 5
        
        self.meta_data_table.loc[len(self.meta_data_table.index)] = [
            self.key, table_name, creation_date, last_update, *[None] * empty
        ]

        if self.debug:
            self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'path'] = test_file

    def get_summary_template(self):
        self.template, self.prefix, self.suffix, self.examples = load_templates('summary_template')
        self.prompt_template = get_template(self.examples,
                                            self.prefix,
                                            self.suffix)

    def get_summary(self):
        self.get_summary_template()
        column_names: list = ast.literal_eval(self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'column_names'][0])
        chain = self.prompt_template | ChatOpenAI(temperature=0.0,
                                                  openai_api_key=self.openai_api_key,
                                                  model_name="gpt-4-1106-preview")

        answer = chain.invoke({"column_names": column_names,
                               "dataframe_title": self.meta_data_table.loc[self.meta_data_table['key'] ==
                                                                           self.key, 'table_name'].values[0]})

        extracted_answer = answer.content.split('=')[-1][1:]
        description = extracted_answer.split('"')
        self.description = description[int((len(description) + 1) / 2) - 1]
        str_dict = ', '.join(answer.content.split('=')[3].strip().split(',')[:-1]).strip()[1:-1]

        if len(str_dict.split('}')) == 2 and len(str_dict.split('{')) == 2:
            self.keywords = list(ast.literal_eval(', '.join(answer.content.split('=')[3].strip().split(',')[:-1]).strip()[1:-1]).keys())
            self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'description'] = self.description
            self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'keywords'] = f'{self.keywords}'
        else:
            self.get_summary()

    def columns_info(self) -> None:
        """
        Inserts the column names and types.
        if type == object: needs to be updated if customer wants.
        if no column header - a new one will be created and inserted.
        :return: column names and types
        """
        column_description = self.get_column_description()
        self.column_names = ast.literal_eval(column_description)
        self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'column_names'] = f'{self.column_names}'

    def get_column_types(self):
        """
        Sets the column types
        """
        self.table = self.table.dropna()
        self.table = self.table.replace([np.inf, -np.inf], 0)
        x = self.table.infer_objects().dtypes
        for i, col in enumerate(self.table.columns):
            if x[col] == 'object':
                if np.all([isinstance(x, str) for x in self.table[col].tolist()]):
                    if np.all([bool(re.search(r'\d', x)) for x in self.table[col].tolist()]):
                        if not np.any([is_date(x) for x in self.table[col].tolist()]):
                            if not np.any([is_range(x) for x in self.table[col].tolist()]):
                                if not np.any([special_char(x) for x in self.table[col].tolist()]):
                                    if len(parse_string(self.table[col].iloc[0])['others_list']) < 3:
                                        if check_number_unit(self.table[col].iloc[0], Units):
                                            val_units_df = self.table[col].apply(lambda x: parse_number(**parse_string(x)))
                                            df_new = pd.DataFrame.from_records(val_units_df.to_list(), columns=[f'{col}', 'unit'])
                                            self.table.drop(columns=col, inplace=True)
                                            self.table = pd.concat([self.table, df_new], axis=1)
                else:
                    if np.all([isinstance(x, int) for x in self.table[col].tolist()]):
                        self.table.astype({col: 'int'})

        self.col_types = self.table.infer_objects().dtypes
        self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'column_types'] = \
            f'{[str(ct) for ct in self.col_types.to_list()]}'

    def get_columns_template(self):
        self.template, self.prefix, self.suffix, self.examples = load_templates('columns_template')
        self.prompt_template = get_template(self.examples,
                                            self.prefix,
                                            self.suffix)

    def get_unknown_column(self, column_names, col_vals):
        self.get_columns_template()
        chain = self.prompt_template | ChatOpenAI(temperature=0.2,
                                                  openai_api_key=self.openai_api_key,
                                                  model_name="gpt-4-1106-preview")

        answer = chain.invoke({"Current_column_name": column_names, "Cell_Values": col_vals})
        return ast.literal_eval(answer.content)

    def get_column_description(self) -> str:
        """
        Via Completion (gpt) estimates the meaning of each column
        :return: a list of column names
        """

        column_descriptions = []
        table_cols = [self.random_selection(self.table.iloc[:, i].tolist(), np.min(np.array([10, len(self.table.index)]))) for i in range(len(self.table.columns))]

        for i, col in enumerate(table_cols):
            current_col_name = self.clean_results(self.table.columns.tolist()[i])
            if self.get_unknown_column(current_col_name, col):
                task = (
                    f'We have a data table that has some undefined column names, like "Unnamed: 0" or "Unnamed:'
                    f'We need to replace these names with useful names.'
                    f'Describe what this table column is about in one or two words.'
                    f'As a context I give you a list of all current table names in order: {self.table.columns.tolist()}'
                    f'Only return your result. No explanations! Example output: "Customer Feedback"')

                name_response = openai.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=task,
                    temperature=0.2,
                    max_tokens=150,
                )

                res = name_response.choices[0].text
                column_descriptions.append(self.clean_results(res))
            else:
                column_descriptions.append(current_col_name)

        return f'{column_descriptions}'

    @staticmethod
    def clean_results(result):
        clean_str = re.sub('\s+', ' ', result)
        clean_str = clean_str.strip()
        clean_str = clean_str.replace('"', '')
        return clean_str.replace('_', ' ')

    @staticmethod
    def random_selection(arr, n):
        return np.random.choice(arr, n)

    def date_format(self, idx: int) -> None:
        """
        Detects datetime in strings and updates its format
        it makes a guess if the date is ambiguous
        optional dayfirst = True
        :param idx: index referencing a column
        :return: updates datetime formate from string, type: None
        """
        parser = partial(dparser.parse, fuzzy=True)
        self.table.iloc[:, idx] = self.table.iloc[:, idx].apply(parser)

    def get_context_template(self):
        self.template, self.prefix, self.suffix, self.examples = load_templates('context_template')
        self.prompt_template = get_template(self.examples,
                                            self.prefix,
                                            self.suffix)

    def get_context(self) -> str:
        """
        Via Completion (gpt) estimates context of given table.
        :return: context value
        """
        self.get_context_template()

        description = self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'description']
        context_lib = [item.value for item in WordContext]
        exceptions = [x.value for x in WordException]

        chain = self.prompt_template | ChatOpenAI(temperature=0.0,
                                                  openai_api_key=self.openai_api_key,
                                                  model_name="gpt-4-1106-preview")

        answer = chain.invoke({"summary": description,
                               "options": context_lib,
                               "exceptions": exceptions})

        match = re.search(r'\bbusiness area\b', answer.content)
        pos = match.span()

        context = answer.content[pos[1] + 4:-1]

        return context

    def context(self) -> None:
        """
        Sets up a context value for given table.
        """
        self.context_result = self.get_context()

        regex = r'\b\w+\b'
        words = re.findall(regex, self.context_result)
        context_result = ' '.join([w.lower() for w in words])

        if context_result not in [item.value for item in WordContext]:
            extend_enum(WordContext, f'{context_result}', context_result)

        self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'context'] = context_result
        self.save_WordContext()

    def scopes(self) -> None:
        """
        Extracts for every column all scopes of given table.
        """
        scopes = dict()
        for idx, col in enumerate(self.table.columns):
            if (self.table.iloc[:, idx].dtypes == 'object') or (self.table.iloc[:, idx].dtypes == 'StringDtype'):
                scopes[idx] = self.table.iloc[:, idx].unique().tolist()
            elif self.table.iloc[:, idx].dtypes in ['int', 'float']:
                scopes[idx] = [self.table.iloc[:, idx].min(), self.table.iloc[:, idx].max()]
            elif (self.table.iloc[:, idx].dtypes == np.dtype('datetime64[ns]')) or (
                    self.table.iloc[:, idx].dtypes == np.dtype('<M8[ns]')):
                scopes[idx] = [self.table.iloc[:, idx].min().strftime('%d-%m-%Y, %H:%M:%S'),
                               self.table.iloc[:, idx].max().strftime('%d-%m-%Y, %H:%M:%S')]

        self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'scopes'] = str(scopes)

    @staticmethod
    def good_answer(context_result: str) -> bool:
        """
        Checks if llm anser is acceptable.
        :param context_result: Answer of llm
        :return: True if acceptable else False
        """
        exceptions = [x.value for x in WordException]

        regex = r'\b\w+\b'
        words = re.findall(regex, context_result)

        for word in words:
            if word.lower() in exceptions:
                return False

        return True

    @staticmethod
    def save_WordContext():
        """
        Saves all context attributes into yaml file.
        """
        script_path = Path(__file__).parent.resolve()
        WordContext_dict = {i.name: i.value for i in WordContext}
        with open(f'{script_path}/utils/WordContext.yaml', 'w', ) as f:
            yaml.dump(WordContext_dict, f, sort_keys=False)

    @staticmethod
    def load_WordContext():
        """
        Loads context attributes into enum class named WordContext.
        """
        cwd = Path(__file__).resolve().parents[0]
        if Path(f'{cwd}/utils/WordContext.yaml').is_file():
            with open(f'{cwd}/utils/WordContext.yaml', 'r') as f:
                WordContext_dict = yaml.safe_load(f)

            for key in WordContext_dict:
                if WordContext_dict[key] not in [item.value for item in WordContext]:
                    extend_enum(WordContext, f'{key}', WordContext_dict[key])

    @staticmethod
    def load_Units():
        """
        Loads context attributes into enum class named WordContext.
        """
        cwd = Path(__file__).resolve().parents[0]
        if Path(f'{cwd}/utils/units.yaml').is_file():
            with open(f'{cwd}/utils/units.yaml', 'r') as f:
                unit_dict = yaml.safe_load(f)

            for key in unit_dict:
                if unit_dict[key] not in [item.value for item in Units]:
                    extend_enum(Units, f'{key}', unit_dict[key])

    def load_into_vector_db(self):
        vec_num = self.vs.me.collection.count()
        self.vs.embed_context(vec_num=vec_num, meta_data=self.meta_data_table.loc[self.meta_data_table['key'] == self.key])

    def save_meta_data_table(self) -> None:
        """
        Saves metadata table as json and excel files.
        """
        if self.debug:
            cwd = Path(__file__).resolve().parent
            if os.path.isfile(f'{cwd}/test_files/test_data/quality_data/quality_meta.xlsx'):
                current_meta = pd.read_excel(f'{cwd}/test_files/test_data/quality_data/quality_meta.xlsx',
                                             index_col=None)
                new_meta = pd.concat([current_meta, self.meta_data_table])
                new_meta.to_excel(f'{cwd}/test_files/test_data/quality_data/quality_meta.xlsx', index=False)
            else:
                self.meta_data_table.to_excel(f'{cwd}/test_files/test_data/quality_data/quality_meta.xlsx', index=False)

        else:
            base_url = 'https://backend.vhi.ai/service-api'
            headers = {'Authorization': f'Bearer {self.token}',
                       'x-api-key': f'{self.api_key}'}
            metadata = self.meta_data_table.to_json(path_or_buf=None)
            dict_meta = json.loads(metadata)
            result = requests.patch(f"{base_url}/metadata", headers=headers, json=dict_meta)
            if result.status_code == 200:
                self.common.write(WriteType.RESULT, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set a table.')
    parser.add_argument('-o', '--openai_api_key', help='OPENAI Key', required=True)
    parser.add_argument('-a', '--api_key', help='Backend Key', required=True)
    parser.add_argument('-token', '--token', required=True)
    parser.add_argument('-sheet', '--sheet_id', required=True)
    args = parser.parse_args()

    setter = TableSetter(**vars(args))
    setter.run()
    setter.save_results()
