import ast
import os
import logging
import argparse
from typing import List

from aenum import extend_enum
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import openai
import re
import dateutil.parser as dparser
import yaml

from enum import Enum
from pathlib import Path
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from vector_search import MetaEngine
from utils import load_templates, get_template


logging.basicConfig(filename='prebuilt.log', format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)


class WordContext(Enum):
    pass


class WordException(Enum):
    MANAGEMENT = "management"
    ROW = "row"
    COLUMN = "column"
    DATA = "data"
    ANALYSIS = "analysis"
    ANALYTICS = "analytics"


class TableSetter:

    def __init__(self,
                 api_key: str,
                 dataset_name: str):

        """
        Creates a metadata table.
        :param api_key: key of openai key
        :param dataset_name: name of dataset with file extension, ex.: table.csv
        """

        # openai.api_key = api_key
        self.openai_api_key = api_key
        os.environ['OPENAI_API_KEY'] = api_key
        
        self.dataset_name = dataset_name
        self.original_dir = Path('datasets/original')
        self.destination_dir = Path('datasets/json_data')

        meta_data_table_json = Path('datasets/meta_data_table.json')
        if meta_data_table_json.is_file():
            self.meta_data_table = pd.read_json('datasets/meta_data_table.json')
        else:
            self.meta_data_table = pd.DataFrame(
                columns=['key', 'path', 'table_name', 'creation_date', 'last_update', 'context',
                         'column_names', 'scopes', 'description', 'column_type'])

        self.table_path = self.original_dir / dataset_name

        self.llm = OpenAI(api_token=api_key, engine="gpt-3.5-turbo")
        self.pandas_ai = PandasAI(self.llm, enable_cache=False, verbose=True, conversational=True)

        self.key = None
        self.table = None
        self.destination_name = None
        self.column_description = None

        self.template = None
        self.examples = None
        self.prefix = None
        self.suffix = None

        self.prompt_template = None

        self.terminate = False

        self.load_WordContext()

    def update_table(self):
        """
        If entry in metadata table already exists and just has some changes.
        :return: Updated metadata table for given key of changed table.
        """
        date_time = datetime.fromtimestamp(datetime.timestamp(datetime.now()))
        last_update = date_time.strftime("%d-%m-%Y, %H:%M:%S")

        self.load_table(self.key)
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
        self.scopes()
        self.get_summary()
        if self.terminate:
            return
        self.context()
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

    def load_table(self, key: str) -> None:
        """
        Transforms into pandas frame and json file.
        * issue: if csv has a leading seperator -> creates extra column
        :param key: identifier
        """
        file_extension = self.table_path.suffix

        if file_extension in ['.csv', '.xlsx']:
            self.table = pd.read_csv(self.table_path) if file_extension == '.csv' else pd.read_excel(self.table_path)
            self.table.to_json(self.destination_dir / f'{key}.json')

        else:
            self.table = pd.read_json(self.table_path)

    def setup_entry(self) -> None:
        """
        Creates new entry in meta_data_table w/ key, table_name, creation_date, last_update
        """
        key = datetime.timestamp(datetime.now())
        table_name = self.destination_name if self.destination_name else Path(self.dataset_name).stem
        date_time = datetime.fromtimestamp(key)
        last_update = creation_date = date_time.strftime("%d-%m-%Y, %H:%M:%S")
        self.key = str(key).replace('.', '_')

        self.load_table(self.key)
        self.meta_data_table.loc[len(self.meta_data_table.index)] = [
            self.key, str(self.table_path), table_name, creation_date, last_update, *[None]*5
        ]

    def get_summary_template(self):
        self.template, self.prefix, self.suffix, self.examples = load_templates('summary_template')
        self.prompt_template = get_template(self.template,
                                            self.examples,
                                            self.prefix,
                                            self.suffix,
                                            ["column_names", "dataframe_title"])

    def get_summary(self):
        self.get_summary_template()

        key_words: list = self.table.columns.tolist()
        exceptions = [x.value for x in WordException]

        from langchain.llms import OpenAI

        openai = OpenAI(
            model_name="text-davinci-003",
            openai_api_key=self.openai_api_key
        )

        completion = openai(
            self.prompt_template.format(
                column_names=key_words,
                dataframe_title=self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'table_name'].values[0])
        )

        print(completion)

        match = re.search(r'\bdescription\b', completion)
        try:
            pos = match.span()
        except:
            logging.debug(f'GPT answer: {completion}')
            pos = [0, 0]

        description = completion[pos[1]+4:-1]
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
        self.update_column_types()
        column_type = str({i: t.name for i, t in enumerate(self.table.dtypes.tolist())})

        if self.table.columns.tolist()[0] not in list(set(self.table.iloc[:, 0])):
            column_names = str({i: e for i, e in enumerate(self.table.columns.values.tolist())})
        else:
            self.table.columns = column_names = self.column_description
            self.table.loc[len(self.table.index)] = self.table.columns.tolist()

        self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'column_names'] = column_names
        self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'column_type'] = column_type

    def get_column_description(self) -> str:
        """
        Via Completion (gpt) estimates the meaning of each column
        :return: a list of column names
        """
        name_prompt = (f'Describe every child list in this list: '
                       f'{[self.table.iloc[:, i].tolist()[:np.min(np.array([10, len(self.table.index)]))] for i in range(len(self.table.columns))]} '
                       f'in one or two words.')

        name_response = openai.Completion.create(
            engine="text-davinci-003",
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

    def update_column_types(self) -> None:
        """
        Checks if column type is string and if this string contains numbers and symbols to find date times and distances
        :param idx: column
        :return: updates formats of date times and distances in strings
        """
        for idx, col in enumerate(self.column_description):
            if isinstance(self.table.iloc[:, idx].tolist()[0], str):
                if bool(re.search(r'\d', self.table.iloc[:, idx].tolist()[0])):
                    try:
                        self.date_format(idx)
                    except:
                        if self.unit_checker(idx):
                            self.distance_format(idx)

    def get_context_template(self):
        self.template, self.prefix, self.suffix, self.examples = load_templates('context_template')
        self.prompt_template = get_template(self.template,
                                            self.examples,
                                            self.prefix,
                                            self.suffix,
                                            ["summary", "options"])

    def get_context(self) -> str:
        """
        Via Completion (gpt) estimates context of given table.
        :return: context value
        """
        self.get_context_template()

        description = self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'description']
        exceptions = [x.value for x in WordException]
        context_lib = [item.value for item in WordContext]

        from langchain.llms import OpenAI

        openai = OpenAI(
            model_name="text-davinci-003",
            openai_api_key=self.openai_api_key
        )

        completion = openai(
            self.prompt_template.format(
                summary=description,
                options=context_lib)
        )

        print(completion)

        match = re.search(r'\bbusiness area\b', completion)
        pos = match.span()

        context = completion[pos[1] + 4:-1]

        return context

    def context(self) -> None:
        """
        Sets up a context value for given table.
        """
        context_result = self.get_context()
        while not self.good_answer(context_result):
            # TODO: set condition to break - how many loops ?
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
            elif (self.table.iloc[:, idx].dtypes == np.dtype('datetime64[ns]')) or (self.table.iloc[:, idx].dtypes == np.dtype('<M8[ns]')):
                scopes[idx] = [self.table.iloc[:, idx].min().strftime('%d-%m-%Y, %H:%M:%S'), self.table.iloc[:, idx].max().strftime('%d-%m-%Y, %H:%M:%S')]

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
        WordContext_dict = {i.name: i.value for i in WordContext}
        with open('./WordContext.yaml', 'w', ) as f:
            yaml.dump(WordContext_dict, f, sort_keys=False)

    @staticmethod
    def load_WordContext():
        """
        Loads context attributes into enum class named WordContext.
        """
        if Path('WordContext.yaml').is_file():
            with open('WordContext.yaml', 'r') as f:
                WordContext_dict = yaml.safe_load(f)

            for key in WordContext_dict:
                if WordContext_dict[key] not in [item.value for item in WordContext]:
                    extend_enum(WordContext, f'{key}', WordContext_dict[key])

    def save_meta_data_table(self) -> None:
        """
        Saves metadata table as json and excel files.
        """
        self.meta_data_table.to_json('datasets/meta_data_table.json')
        self.meta_data_table.to_excel('datasets/meta_data_table.xlsx', index=False)


class Extractor:

    def __init__(self,
                 openai_api_key: str,
                 customer_request: str,
                 selected_tables: List[str] = None):

        os.environ['OPENAI_API_KEY'] = openai_api_key

        self.openai_api_key = openai_api_key
        self.customer_request = customer_request
        self.llm = OpenAI(api_token=openai_api_key, model="gpt-4", max_tokens=1000)
        self.pandas_ai = PandasAI(self.llm, enable_cache=True, verbose=True, conversational=True)

        self.meta_data_table = pd.read_json('datasets/meta_data_table.json')
        if selected_tables:
            self.get_selected_tables(selected_tables)
        self.load_WordContext()

        self.prompt_template = None

        self.template = None
        self.examples = None
        self.prefix = None
        self.suffix = None

        self.keys_words = None

        self.selected_table_keys = []
        self.selected_tables = []

        # save file name for each table

    def get_selected_tables(self, selected_tables):
        import copy
        selected_df = list()
        for name in selected_tables:
            table = self.meta_data_table.loc[self.meta_data_table['table_name'] == name]
            selected_df.append(copy.deepcopy(table))

        self.meta_data_table = pd.concat(selected_df)
        self.meta_data_table.reset_index(drop=True, inplace=True)

    def get_meta_template(self):
        self.template, self.prefix, self.suffix, self.examples = load_templates('meta_template')
        self.prompt_template = get_template(self.template,
                                            self.examples,
                                            self.prefix,
                                            self.suffix,
                                            ["question"])

    def key_word_selection(self):
        prompt = self.prompt_template.format(question=self.customer_request)
        prompt_template = ChatPromptTemplate.from_template(prompt)
        message = prompt_template.format_messages()

        llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv(self.openai_api_key))
        response = llm(message)

        self.keys_words = ast.literal_eval(response.content)

    def select_tables(self):
        meta_columns = [list(ast.literal_eval(self.meta_data_table['column_names'][i]).values()) for i in range(len(self.meta_data_table))]
        table_indices = [self.meta_data_table['key'][i] for i in range(len(self.meta_data_table))]

        me = MetaEngine('test_collection')

        vec_num = 0

        for col, table_index in zip(meta_columns, table_indices):
            me.embed_new_vec(vec_num, table_index, col)
            vec_num += len(col)

        me.load_vec()

        temp_res = []

        for i, query in enumerate(self.keys_words):
            similar_results = me.find_semantic(query)
            for res in similar_results:
                if i == 0:
                    self.selected_table_keys.append(res[0])
                else:
                    temp_res.append(res[0])

            # to get an intersection of all values
            if i != 0:
                self.selected_table_keys = [value for value in temp_res if value in self.selected_table_keys]

        self.selected_table_keys = list(set(self.selected_table_keys))
        self.get_tables()
        self.run_request()

    def get_tables(self):
        logging.info(f"SELECTED TABLES")
        pd.set_option('display.max_columns', None)

        for key in self.selected_table_keys:
            table_path = Path(self.meta_data_table.loc[self.meta_data_table['key'] == key, 'path'].tolist()[0])
            file_extension = table_path.suffix

            if file_extension in ['.csv', '.xlsx']:
                table = pd.read_csv(table_path) if file_extension == '.csv' else pd.read_excel(table_path)

                logging.info(f"\n{table.head(n=3)}")

                self.selected_tables.append(table)

    def run_request(self):
        respond = self.pandas_ai.run(data_frame=self.selected_tables, prompt=self.customer_request, anonymize_df=True, show_code=True)
        print(respond)
        # print(self.pandas_ai.logs)
        # print(self.pandas_ai.logs[-4]['msg'])

    def run_request_with_1_df(self):
        from llama_index.query_engine.pandas_query_engine import PandasQueryEngine

        query_engine = PandasQueryEngine(df=self.selected_tables[0], verbose=True)

        response = query_engine.query(
            self.customer_request
        )

    @staticmethod
    def load_WordContext():
        """
        Loads context attributes into enum class named WordContext.
        """
        if Path('WordContext.yaml').is_file():
            with open('WordContext.yaml', 'r') as f:
                WordContext_dict = yaml.safe_load(f)

            for key in WordContext_dict:
                if WordContext_dict[key] not in [item.value for item in WordContext]:
                    extend_enum(WordContext, f'{key}', WordContext_dict[key])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-o', '--openai_api_key', help='OPENAI Key', required=True)
    parser.add_argument('-r', '--customer_request', help='Task for AI.', required=True, nargs='+', dest='customer_request')
    parser.add_argument('--selected_tables', help='A list of table names that should be selected.')
    args = parser.parse_args()

    extraction = Extractor(**vars(args))
    extraction.get_meta_template()
    extraction.key_word_selection()
    extraction.select_tables()
