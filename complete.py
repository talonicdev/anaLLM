import argparse
import os
import re
import os.path
import copy
import requests
from io import StringIO
from typing import Tuple

import pandas as pd
from decouple import config

from langchain_community.chat_models import ChatOpenAI
from pandasai.llm import OpenAI

from prebuilt import Extractor
from utils.conf import load_templates, get_template

from config import Config
from common import Common, Requests, WriteType

os.environ['API_USER'] = config('USER')
os.environ['OPENAI_API_KEY'] = config('KEY')


class CompleteTable:

    def __init__(self,
                 openai_api_key: str,
                 api_key: str,
                 token: str,
                 sheet_id: str = None,
                 content: str = None,
                 table_path: str = None,
                 users_prompt: str = None,
                 debug: bool = False,
                 ):

        self.config = Config(
            token = token,
            api_key = api_key,
            sheet_id = sheet_id,
            openai_api_key = openai_api_key
        )
        self.requests = Requests(config=self.config)
        self.common = Common(config=self.config)
        
        self.content = content
        self.table_path = table_path
        self.openai_api_key = openai_api_key
        self.api_key = api_key
        self.token = token
        self.debug = debug
        self.users_prompt = users_prompt

        self.template = None
        self.examples = None
        self.prefix = None
        self.suffix = None
        self.table = None

        self.prompt_template = None
        self.load_table(api_key, token, sheet_id)

    def call_table(self, sheet_id):
        response = self.requests.get(f'sheet/{sheet_id}')
        if response.status_code == 200:
            sheet_data = response.json()
            columns = sheet_data['sheet'].keys()
            self.table = pd.DataFrame(sheet_data['sheet'], columns=columns)
            self.common.write(WriteType.DEBUG,self.table)
        else:
            self.common.write(WriteType.ERROR,{'sheet_id':sheet_id,'code':response.status_code,'text':response.text})

    def load_table(self, api_key, token, sheet_id):
        if self.content:
            self.table = pd.read_csv(StringIO(self.content))
            self.table = self.table[1:]
        elif self.table_path:
            self.table = pd.read_excel(self.table_path, header=1)
        else:
            if self.debug:
                # response is a list of dictionaries, where each dictionary is representative for a table
                response = self.requests.get('sheet-overview')
                if response.status_code == 200:
                    all_sheets = response.json()
                    example_sheet_id = all_sheets[0]['sheetId']
                    self.call_table(example_sheet_id)
                elif response.status_code == 401:
                    raise ValueError("Invalid token.")
                else:
                    print("Error:", response.status_code, response.text)

            else:
                self.call_table(sheet_id)

    def get_empty_cols(self) -> Tuple[list, ...]:
        """
        Finds the empty columns to be filled.
        :return: name of columns, changed (lower case) and original
        """
        empty_cols = []
        original_cols = []
        for col in self.table.columns:
            if self.table[col].isnull().sum().sum() == self.table.shape[0]:
                empty_cols.append(col.lower())
                original_cols.append(col)

        return empty_cols, original_cols

    def get_complete_template(self):
        """
        Load the complete template.
        """
        self.template, self.prefix, self.suffix, self.examples = load_templates('complete_template')
        self.prompt_template = get_template(self.examples,
                                            self.prefix,
                                            self.suffix)

    def get_rating_template(self):
        """
        Load the complete template.
        """
        self.template, self.prefix, self.suffix, self.examples = load_templates('rating_template')
        self.prompt_template = get_template(self.examples,
                                            self.prefix,
                                            self.suffix)

    def get_table_question(self,
                           empty_cols: list,
                           original_cols: list):
        """
        Generates a request based on the tables that is given.
        :param empty_cols: List of empty columns. (lower case letters)
        :param original_cols: List of empty columns.
        :return: request, useful columns
        """

        self.get_complete_template()
        exists_cols = copy.deepcopy(self.table.columns.tolist())
        for i in range(len(original_cols)):
            idx = exists_cols.index(original_cols[i])
            exists_cols.pop(idx)

        chain = self.prompt_template | ChatOpenAI(temperature=0.6,
                                                  openai_api_key=self.openai_api_key,
                                                  model_name="gpt-4-1106-preview")

        answer = chain.invoke({"new_column": empty_cols,
                               "question": self.users_prompt,
                               "exists": exists_cols})

        match = re.search(r'\bUseful columns\b', answer.content)
        pos2 = match.span()
        useful = answer.content[pos2[1] + 4:-1]
        useful = [s.strip().strip("'") for s in useful.split(',')]

        match = re.search(r'\brequest\b', answer.content)
        pos1 = match.span()
        request = answer.content[pos1[1] + 3: pos2[0]]

        return request, useful, exists_cols, original_cols

    def get_rating_check(self,
                         empty_cols: list,
                         original_cols: list):

        self.get_rating_template()

        exists_cols = copy.deepcopy(self.table.columns.tolist())
        for i in range(len(original_cols)):
            idx = exists_cols.index(original_cols[i])
            exists_cols.pop(idx)

        chain = self.prompt_template | ChatOpenAI(temperature=0.6,
                                                  openai_api_key=self.openai_api_key,
                                                  model_name="gpt-4-1106-preview")

        answer = chain.invoke({"new_column": empty_cols,
                               "exists": exists_cols})

        match = re.search(r'\bUseful columns\b', answer.content)
        pos2 = match.span()
        useful = answer.content[pos2[1] + 4:-1]
        useful = [s.strip().strip("'") for s in useful.split(',')]

        match = re.search(r'\brequest\b', answer.content)
        pos1 = match.span()
        request = answer.content[pos1[1] + 3: pos2[0]]

        return request, useful, exists_cols, original_cols

    def create_table(self,
                     request: str,
                     useful: list,
                     exists_cols: list,
                     empty_cols: list):
        """
        Fills the empty columns from the given table.
        :param request: request
        :param useful: useful columns
        """
        import uuid

        add_txt = ''
        add_words = []
        from pandas.api.types import is_string_dtype
        for col in exists_cols:
            if (is_string_dtype(self.table[col])) and not (self.table[col].str.contains('.*[0-9].*', regex=True).any()):
                add_words.extend(self.table[col].unique().tolist())

                obj_elem = ', '.join(("'" + f"{e}" + "'" for e in self.table[col].unique().tolist()))
                add_txt += f' The {col} are: {obj_elem}'
        request += add_txt

        #df = self.table.drop(columns=empty_cols, inplace=False)
        df = self.table

        extraction = Extractor(openai_api_key=self.openai_api_key,
                               customer_request=request,
                               api_key=self.api_key,
                               token=self.token)

        extraction.get_meta_template()
        extraction.key_word_selection()
        extraction.keys_words += add_words
        extraction.select_tables()
        extraction.selected_tables.append(df)
        extraction.run_request()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-o', '--openai_api_key', help='OPENAI Key', required=True)
    parser.add_argument('-a', '--api_key', help='Backend Key', required=True)
    parser.add_argument('-token', '--token', required=True)
    parser.add_argument('-sheet', '--sheet_id', required=True)
    parser.add_argument('-query', '--users_prompt')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-p', '--table_path', help='Path of the table to be filled.')
    group.add_argument('-c', '--content', help='Table content as buffer / str to be filled.')
    args = parser.parse_args()

    ct = CompleteTable(**vars(args))
    e_cols, o_cols = ct.get_empty_cols()
    c_request, useful, exists_cols, empty_cols = ct.get_table_question(e_cols, o_cols)
    ct.create_table(c_request, useful, exists_cols, empty_cols)
