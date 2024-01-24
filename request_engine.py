import ast
import os
import logging

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

from utils.conf import load_templates, get_template, WordContext, WordException
from vector_search import MetaEngine

logging.basicConfig(filename='prebuilt.log',
                    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


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
            self.key, str(self.table_path), table_name, creation_date, last_update, *[None] * 3
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
                dataframe_title=self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'table_name'].values[
                    0])
        )

        match = re.search(r'\bdescription\b', completion)
        try:
            pos = match.span()
        except:
            logging.debug(f'GPT answer: {completion}')
            pos = [0, 0]

        description = completion[pos[1] + 4:-1]
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
        WordContext_dict = {i.name: i.value for i in WordContext}
        with open('utils/WordContext.yaml', 'w', ) as f:
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
        me = MetaEngine()

        meta_columns = [list(ast.literal_eval(self.meta_data_table['column_names'][i]).values()) for i in range(len(self.meta_data_table))]
        table_indices = [self.meta_data_table['key'][i] for i in range(len(self.meta_data_table))]

        me.create_new_collection('default_collection')
        vec_num = 0
        for col, table_index in zip(meta_columns, table_indices):
            me.embed_new_vec(vec_num, table_index, col)
            vec_num += len(col)

    def save_meta_data_table(self) -> None:
        """
        Saves metadata table as json and excel files.
        """
        self.meta_data_table.to_json('datasets/meta_data_table.json')
        self.meta_data_table.to_excel('datasets/meta_data_table.xlsx', index=False)
