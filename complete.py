import os
import calendar
import time
import glob, os.path

import pandas as pd
from decouple import config

from pandasai import SmartDataframe
from pandasai import SmartDatalake
from pandasai import Agent
from pandasai.llm import OpenAI

from prebuilt import Extractor
from utils import load_templates, get_template

os.environ['API_USER'] = config('USER')
os.environ['OPENAI_API_KEY'] = config('KEY')


class CompleteTable:

    def __init__(self,
                 table_path: str,
                 openai_api_key: str):

        self.table_path = table_path

        # self.table = pd.read_json(table_path)
        self.table = pd.read_excel(table_path)
        self.table.columns = self.table.iloc[0]
        self.table = self.table[1:]

        self.llm = OpenAI(api_token=openai_api_key, model="gpt-4", max_tokens=1000)

    def get_empty_cols(self):
        empty_cols = []
        original_cols =[]
        for col in self.table.columns:
            if self.table[col].isnull().sum().sum() == self.table.shape[0]:
                empty_cols.append(col.lower())
                original_cols.append(col)

        self.create_table(empty_cols, original_cols)

    def create_table(self, empty_cols: list, original_cols: list):
        import uuid

        txt = (f"{e}" for e in empty_cols)
        request = ', '.join(txt)
        obj_elem = ', '.join(("'" + f"{e}" + "'" for e in self.table.iloc[:, 0].to_list()))
        request = f"Calculate the " + f"{request}" + f" for {obj_elem} ."

        extraction = Extractor(os.environ['KEY'], request)

        extraction.get_meta_template()
        extraction.key_word_selection()
        extraction.select_tables()
        extraction.run_request()

        res_df = pd.merge(self.table, extraction.response, on=self.table.columns[0])
        res_df.drop(columns=original_cols, inplace=True)
        res_df.to_csv(f'./output_tables/filled_table_{uuid.uuid1()}.csv')
        # extraction.clean()

    @staticmethod
    def load_results():
        dfs = []
        filelist = glob.glob(os.path.join('./output_plot', "*.csv"))
        for f in filelist:
            dfs.append(pd.read_csv(f))
        return dfs

    @staticmethod
    def clean_out_dirs(dir_path, file_type):
        filelist = glob.glob(os.path.join(dir_path, file_type))
        for f in filelist:
            os.remove(f)


if __name__ == "__main__":
    # df = pd.read_csv("table")
    ct = CompleteTable('./ex1.xlsx', os.environ['KEY'])
    ct.get_empty_cols()


