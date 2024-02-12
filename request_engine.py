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
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

from utils.conf import load_templates, get_template, WordContext, WordException
from vector_search import MetaEngine

logging.basicConfig(filename='prebuilt.log',
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
                 new_collection=False,
                 debug=False):

        """
        Creates a metadata table.
        :param api_key: key of openai key
        :param dataset_name: name of dataset with file extension, ex.: table.csv
        """

        self.api_key = api_key
        self.token = token
        self.sheet_id = sheet_id
        self.openai_api_key = openai_api_key
        self.new_collection = new_collection
        os.environ['OPENAI_API_KEY'] = openai_api_key

        self.meta_data_table = pd.DataFrame(
            columns=['key', 'table_name', 'creation_date', 'last_update', 'context',
                     'column_names', 'description'])

        self.llm = OpenAI(api_token=openai_api_key, engine="gpt-4-1106-preview")

        self.key = None
        self.table = None
        self.dataset_name = None
        self.destination_name = None
        self.column_description = None

        self.template = None
        self.examples = None
        self.prefix = None
        self.suffix = None
        self.debug = debug

        self.prompt_template = None

        self.terminate = False

        self.load_WordContext()

    def call_table(self, base_url, sheet_id, headers):
        response = requests.get(f"{base_url}/sheet/{sheet_id}", headers=headers)
        if response.status_code == 200:
            sheet_data = response.json()
            self.dataset_name = sheet_data['sheetName'] if 'sheetName' in sheet_data else sheet_data['tableName']
            self.table = pd.DataFrame(sheet_data['sheet'])
        else:
            print("Error:", response.status_code, response.text)

    def load_table(self):
        base_url = 'https://backend.vhi.ai/service-api'
        headers = {'Authorization': f'Bearer {self.token}',
                   'x-api-key': f'{self.api_key}'}

        if self.debug:
            response = requests.get(f"{base_url}/sheet-overview", headers=headers)
            if response.status_code == 200:
                all_sheets = response.json()
                example_sheet_id = all_sheets[0]['sheetId']
                self.call_table(base_url, example_sheet_id, headers)
            elif response.status_code == 401:
                raise ValueError("Invalid token.")
            else:
                print("Error:", response.status_code, response.text)

        else:
            self.call_table(base_url, self.sheet_id, headers)

    def update_table(self):
        """
        If entry in metadata table already exists and just has some changes.
        :return: Updated metadata table for given key of changed table.
        """
        date_time = datetime.fromtimestamp(datetime.timestamp(datetime.now()))
        last_update = date_time.strftime("%d-%m-%Y, %H:%M:%S")

        self.load_table()
        self.table.loc[self.table['key'] == self.key, 'last_update'] = last_update

    def initialise_table(self):
        """
        Initialises a new entry in metadata table.
        """
        self.setup_entry()

    def process_entries(self):
        """
        Inserts values for all columns of row entry with given key.
        """
        self.columns_info()
        self.get_summary()
        if self.terminate:
            return
        self.context()
        self.load_into_vector_db()
        self.save_meta_data_table()

    def run(self, destination_name: str = None, table_key: str = None):
        """
        Updates or initialises entry in metadata table.
        :param destination_name: new short name for given table
        :param table_key: if entry already exists this key is a reference to given table.
        """
        # error handling for wrong key etc...
        if table_key:
            self.key = table_key
            self.update_table()

        else:
            if destination_name:
                self.destination_name = destination_name
            self.initialise_table()

        self.process_entries()

    def setup_entry(self) -> None:
        """
        Creates new entry in meta_data_table w/ key, table_name, creation_date, last_update
        """
        key = datetime.timestamp(datetime.now())
        table_name = self.destination_name if self.destination_name else self.dataset_name
        date_time = datetime.fromtimestamp(key)
        last_update = creation_date = date_time.strftime("%d-%m-%Y, %H:%M:%S")
        self.key = self.sheet_id

        self.load_table()
        self.meta_data_table.loc[len(self.meta_data_table.index)] = [
            self.key, table_name, creation_date, last_update, *[None] * 3
        ]

    def get_summary_template(self):
        self.template, self.prefix, self.suffix, self.examples = load_templates('summary_template')
        self.prompt_template = get_template(self.examples,
                                            self.prefix,
                                            self.suffix)

    def get_summary(self):
        self.get_summary_template()
        key_words: list = self.table.columns.tolist()
        chain = self.prompt_template | ChatOpenAI(temperature=0.0,
                                                  openai_api_key=self.openai_api_key,
                                                  model_name="gpt-4-1106-preview")

        answer = chain.invoke({"column_names": key_words,
                               "dataframe_title": self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'table_name'].values[0]})

        match = re.search(r'\bdescription\b', answer.content)
        try:
            pos = match.span()
        except:
            logging.debug(f'GPT answer: {answer}')
            pos = [0, 0]

        description = answer.content[pos[1] + 4:-1]
        self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'description'] = description

    def columns_info(self) -> None:
        """
        Inserts the column names and types.
        if type == object: needs to be updated if customer wants.
        if no column header - a new one will be created and inserted.
        :return: column names and types
        """

        # update types
        self.column_description = [x.strip() for x in self.get_column_description()[2:].split(',')]

        if self.table.columns.tolist()[0] not in list(set(self.table.iloc[:, 0])):
            column_names = str({i: e for i, e in enumerate(self.table.columns.values.tolist())})
        else:
            self.table.columns = column_names = self.column_description
            self.table.loc[len(self.table.index)] = self.table.columns.tolist()

        self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'column_names'] = column_names

    def get_column_description(self) -> str:
        """
        Via Completion (gpt) estimates the meaning of each column
        :return: a list of column names
        """
        name_prompt = (f'Describe every child list in this list: '
                       f'{[self.table.iloc[:, i].tolist()[:np.min(np.array([10, len(self.table.index)]))] for i in range(len(self.table.columns))]} '
                       f'in one or two words.')

        name_response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=name_prompt,
            temperature=0.6,
            max_tokens=150,
        )

        return name_response.choices[0].text

    def unit_checker(self, idx: int):
        """
        Checks if column has numeric value with unit.
        :param idx: column index
        """
        first_elem_is_num = type(list(self.table.iloc[:, idx].tolist()[0])[0]) in [int, float]
        unit_has_num = bool(re.search(r'\d', ' '.join(list(self.table.iloc[:, idx].tolist()[0])[1:])))

        if first_elem_is_num and not unit_has_num:
            return True
        else:
            return False

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

    def distance_format(self, idx: int) -> None:
        """
        Assumes column format to be : x unit, where x in |R
        changes into : x and adds a new column with the unit
        :param idx: index referencing a column
        :return: type: None
        """
        name = self.table.columns[idx]
        new_col = {f'unit_{name}': self.table.iloc[:, idx].apply(lambda x: x.split(' ')[1])}
        self.table = self.table.assign(**new_col)
        self.table.iloc[:, idx] = self.table.iloc[:, idx].apply(lambda x: float(x.split(' ')[0]))

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
        context_result = self.get_context()

        regex = r'\b\w+\b'
        words = re.findall(regex, context_result)
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
        if Path('utils/WordContext.yaml').is_file():
            with open('utils/WordContext.yaml', 'r') as f:
                WordContext_dict = yaml.safe_load(f)

            for key in WordContext_dict:
                if WordContext_dict[key] not in [item.value for item in WordContext]:
                    extend_enum(WordContext, f'{key}', WordContext_dict[key])

    def load_into_vector_db(self):
        me = MetaEngine(self.token)

        meta_columns = [list(ast.literal_eval(self.meta_data_table['column_names'][i]).values()) for i in range(len(self.meta_data_table))]
        table_indices = [self.meta_data_table['key'][i] for i in range(len(self.meta_data_table))]

        if self.new_collection:
            me.create_new_collection('talonic_collection')
        else:
            me.load_collection('talonic_collection')
        vec_num = 0
        for col, table_index in zip(meta_columns, table_indices):
            me.embed_new_vec(vec_num, table_index, col)
            vec_num += len(col)

    def save_meta_data_table(self) -> None:
        """
        Saves metadata table as json and excel files.
        """
        base_url = 'https://backend.vhi.ai/service-api'
        headers = {'Authorization': f'Bearer {self.token}',
                   'x-api-key': f'{self.api_key}'}
        metadata = self.meta_data_table.to_json()
        dict_meta = ast.literal_eval(metadata)
        result = requests.patch(f"{base_url}/metadata", headers=headers, json=dict_meta)
        print(f'saved table: {result}')
        '''x = result.status_code
        y = result.json()'''

