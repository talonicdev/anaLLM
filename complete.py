import argparse
import os
import re
import os.path
import copy
from typing import Tuple

import pandas as pd
from decouple import config

import openai
from pandasai.llm import OpenAI

from prebuilt import Extractor
from utils.conf import load_templates, get_template

os.environ['API_USER'] = config('USER')
os.environ['OPENAI_API_KEY'] = config('KEY')


class CompleteTable:

    def __init__(self,
                 table_path: str,
                 openai_api_key: str):

        self.table_path = table_path

        # self.table = pd.read_json(table_path)
        self.table = pd.read_excel(table_path)
        self.table = self.table[1:]

        self.template = None
        self.examples = None
        self.prefix = None
        self.suffix = None

        self.prompt_template = None

        self.llm = OpenAI(api_token=openai_api_key, model="gpt-4", max_tokens=1000)

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
        self.prompt_template = get_template(self.template,
                                            self.examples,
                                            self.prefix,
                                            self.suffix,
                                            ["new_column", "exists"])

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

        idx = self.table.columns.tolist().index(original_cols[0])
        exists_cols = copy.deepcopy(self.table.columns.tolist())
        exists_cols.pop(idx)

        response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=self.prompt_template.format(
                new_column=f"{empty_cols[0]}",
                exists=exists_cols),
            temperature=0.6,
            max_tokens=150,
        )

        res = response.choices[0].text

        match = re.search(r'\bUseful columns\b', res)
        pos2 = match.span()
        useful = res[pos2[1] + 3:]
        useful = [s.strip().strip("'") for s in useful.split(',')]

        match = re.search(r'\brequest\b', res)
        pos1 = match.span()
        request = res[pos1[1] + 3: pos2[0]]

        return request, useful

    def create_table(self,
                     request: str,
                     original_cols: list):
        """
        Fills the empty columns from the given table.
        :param request: request
        :param useful: useful columns
        """
        import uuid

        add_txt = ''
        add_words = []
        from pandas.api.types import is_string_dtype
        for col in self.table.columns:
            if (is_string_dtype(self.table[col])) and not (self.table[col].str.contains('.*[0-9].*', regex=True).any()):
                add_words.extend(self.table[col].unique().tolist())

                obj_elem = ', '.join(("'" + f"{e}" + "'" for e in self.table[col].unique().tolist()))
                add_txt += f' The {col} are: {obj_elem}'
        request += add_txt

        df = self.table.drop(columns=original_cols, inplace=False)

        extraction = Extractor(os.environ['KEY'], request)

        extraction.get_meta_template()
        extraction.key_word_selection()
        extraction.keys_words += add_words
        extraction.select_tables()
        extraction.selected_tables.append(df)
        extraction.run_request()

        extraction.response.to_csv(f'./output_tables/filled_table_{uuid.uuid1()}.csv')
        # extraction.clean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-o', '--openai_api_key', help='OPENAI Key', required=True)
    parser.add_argument('-p', '--table_path', help='Path of the table to be filled.', required=True)
    args = parser.parse_args()

    ct = CompleteTable(**vars(args))
    # ct = CompleteTable('test_files/growth.xlsx', os.environ['KEY'])
    e_cols, o_cols = ct.get_empty_cols()
    c_request, c_useful = ct.get_table_question(e_cols, o_cols)
    ct.create_table(c_request, o_cols)


