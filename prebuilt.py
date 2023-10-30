import ast
import os

from aenum import extend_enum
from datetime import datetime
from functools import partial
from typing import Callable

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

from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from vector_search import MetaEngine


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

        openai.api_key = api_key
        
        self.dataset_name = dataset_name
        self.original_dir = Path('datasets/original')
        self.destination_dir = Path('datasets/json_data')
        self.meta_data_table = pd.read_json('datasets/meta_data_table.json')
        # self.meta_data_table = pd.read_excel('datasets/meta_data_table.xlsx')
        self.table_path = self.original_dir / dataset_name

        self.llm = OpenAI(api_token=api_key, engine="gpt-3.5-turbo")
        self.pandas_ai = PandasAI(self.llm, enable_cache=False, verbose=True, conversational=True)

        self.key = None
        self.table = None
        self.destination_name = None
        self.column_description = None

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
        self.summary()
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
            self.key, table_name, creation_date, last_update, *[None]*5]

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

        # Integers, Packages, Prices, Dates, Dates (2), Countries, Ages, Genders, Devices, Durations (v1)
        # Numbers, Plans, Prices, Dates, Dates (2), Countries, Ages, Genders, Devices, Durations (v2)
        # User ID, Subscription Type, Monthly Revenue, Join Date, Last Payment Date, Country, Age, Gender, Device, Plan Duration

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

    def summary(self) -> None:
        """
        Summarizes the content of given table.
        """

        key_words: list = self.table.columns.tolist()
        exceptions = [x.value for x in WordException]

        info1 = f" A list of column names from a dataset from a company will be given to you."
        info2 = f" This dataset has the following title: {self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'table_name']}."

        condition1 = f" Rules: a) Include as much information as possible in all 4 answers. b) Don't use any column names for any answer."

        condition2 = f" Start your 1. answer with the sentence: 'The dataset contains information related to ...' . "
        condition21 = f" For your 1. answer don't use the words in the following list: {exceptions}."
        condition22 = f" And also don't use the companies name or the title of the dataset for this specific answer."

        condition3 = f" Start your 2. answer with: 'The columns in this dataset can be grouped into several categories based on their similarities: ...' . "
        condition4 = f" Start your 3. answer with: 'The data is about the company, ...' . "

        content1 = f" 1. question: What is a context description for the following column names in this list {key_words}? "
        content2 = f" 2. question: Group the columns of this dataset by similarity. Name only these groups."
        content3 = f" 3. question: Which company is that data about?"

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": info1 + info2 + condition1 + condition2 + condition21 + condition22 + condition3 + condition4},
                {"role": "user", "content": content1 + content2 + content3}
            ]
        )

        description = self.format_answers(completion.choices[0].message.content)
        self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'description'] = description

    def get_context(self) -> str:
        """
        Via Completion (gpt) estimates context of given table.
        :return: context value
        """
        description = self.meta_data_table.loc[self.meta_data_table['key'] == self.key, 'description']
        exceptions = [x.value for x in WordException]
        context_lib = [item.value for item in WordContext]
        prompt = (f"Do not use any of the following words in your answer: {exceptions}."
                  f"If one of the following statements {context_lib} that describes the context of {description} return it otherwise find a new one."
                  f"Choose maximum two words to describe this context and do not use the word data.")

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.6,
            max_tokens=3000,
        )

        # 0 user subscriptions could refer to a management context where a company tracks the number of users
        # who have subscribed to their services or products. This could be used to measure the success of
        # a marketing campaign or to understand the customer base of the company.
        # -> Data Management

        return self.format_answers(response.choices[0].text)

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

    @staticmethod
    def format_answers(answer: str) -> str:
        """
        Adjust llm string answer into acceptable format.
        :param answer: llm answer
        :return: new formatted answer
        """

        start = [(m.start(0), m.end(0)) for m in re.finditer(r"(?<![^\s>])([0-9]+)\. ", answer)]
        answer1 = answer[start[0][1]: start[1][0]].split('.')[0]
        answer2 = answer[start[1][1]: start[2][0]]
        answer3 = answer[start[2][1]:]

        a = 'The dataset contains information related to'
        b = 'The columns in this dataset can be grouped into several categories based on their similarities:'
        c = 'The data is about the company'

        proper_answers = [proper_start.find(ans) for proper_start, ans in zip([answer1, answer2, answer3], [a, b, c])]

        # - User Information: column names (either dot and new sentence or continue) about meaning of each column

        return c + ' ' + a + ' ' + b

    def save_meta_data_table(self) -> None:
        """
        Saves metadata table as json and excel files.
        """
        self.meta_data_table.to_json('datasets/meta_data_table.json')
        self.meta_data_table.to_excel('datasets/meta_data_table.xlsx', index=False)


class Extractor:

    def __init__(self,
                 openai_api_key: str,
                 customer_request: str):

        os.environ['OPENAI_API_KEY'] = openai_api_key

        self.openai_api_key = openai_api_key

        self.customer_request = customer_request
        self.llm = OpenAI(api_token=api_key, model="gpt-4", max_tokens=1000)
        self.pandas_ai = PandasAI(self.llm, enable_cache=False, verbose=True, conversational=True)

        self.meta_data_table = pd.read_json('datasets/meta_data_table.json')

        self.selection_prompt = (f"Each row in the given dataframe corresponds to another dataframe. "
                                 f"Select the keys from the given dataframe that are required to execute the following "
                                 f"request {customer_request}.")
        self.load_WordContext()

        self.prompt_template = None

        self.template = None
        self.examples = None
        self.prefix = None
        self.suffix = None

        self.keys_words = None

    def load_templates(self):
        script_path = Path(__file__).parent.resolve()

        with open(script_path / 'meta_template/example_template.txt', 'r') as file:
            self.template = file.read().replace('\n', ' \n ')

        with open(script_path / 'meta_template/prefix.txt', 'r') as file:
            self.prefix = file.read().replace('\n', ' \n ')

        with open(script_path / 'meta_template/suffix.txt', 'r') as file:
            self.suffix = file.read().replace('\n', ' \n ')

        with open(script_path / 'meta_template/examples.yaml', 'r') as file:
            self.examples = yaml.safe_load(file)

        self.examples = [self.examples[k] for k in self.examples.keys()]

    def get_template(self):
        """
        query_item : rows = user request
        query_entries: columns = add infos
        answer: model answer
        :return:
        """

        example_template = """
        EXAMPLE: 
        Request Item List: {question}
        Answer: {answer}
        """

        example_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template=example_template
        )

        self.prompt_template = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=example_prompt,
            prefix=self.prefix,
            suffix=self.suffix,
            input_variables=["question"],
            example_separator="\n\n"
        )

    def key_word_selection(self):
        prompt = self.prompt_template.format(question=self.customer_request)
        prompt_template = ChatPromptTemplate.from_template(prompt)
        message = prompt_template.format_messages()
        llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv(self.openai_api_key))
        response = llm(message)
        print(response.content)

        self.keys_words = response.content

    def select_tables(self):
        script_path = Path(__file__).parent.resolve()

        meta_table = pd.read_excel(script_path / 'datasets/meta_data_table.xlsx')

        meta_columns = [list(ast.literal_eval(meta_table['column_names'][i]).values()) for i in range(len(meta_table))]
        corpus = [item for row in meta_columns for item in row]

        me = MetaEngine('test_collection')

        for idx, col in enumerate(meta_columns):
            me.embed_new_vec(idx, col)

        # me.load_vec()

        for query in self.keys_words:
            me.find_cos(corpus, query)
            me.find_sem(corpus, query)

    def get_datasets(self):
        rootdir = Path('datasets/original')
        file_list = [f for f in rootdir.glob('**/*') if f.is_file()]

        for file in file_list:
            file_extension = file.suffix
            self.temp_data = pd.read_csv(file) if file_extension == '.csv' else pd.read_excel(file)

            if self.table_relevance():
                if self.data.empty:
                    self.data = self.temp_data
                else:
                    self.data = pd.concat([self.data, self.temp_data], axis=1)

    def table_relevance(self):
        return self.pandas_ai(data_frame=self.meta_data_table,
                              prompt=self.selection_prompt,
                              anonymize_df=True,
                              show_code=False)

    def run_request(self):
        self.pandas_ai(data_frame=self.data, prompt=self.customer_request, anonymize_df=True, show_code=True)

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
    api_key = "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"

    '''ex1 = 'Advertising Budget and Sales.csv'
    ex2 = 'Adidas US Sales Datasets.xlsx'
    ex3 = 'Netflix Userbase.csv'

    setter = TableSetter(api_key, ex2)
    setter.run()'''

    customer_request = "Do females or males generate the highest revenue?"
    extraction = Extractor(api_key, customer_request)

    extraction.load_templates()
    extraction.get_template()
    extraction.key_word_selection()
    extraction.select_tables()

    '''extraction.select_tables()
    extraction.get_datasets()
    extraction.run_request()'''

