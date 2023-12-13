import ast
import os
import logging
import argparse
import calendar
import time
from typing import List
import glob, os.path
from io import StringIO

from aenum import extend_enum

import pandas as pd
import yaml

from pathlib import Path
from pandasai import SmartDataframe
from pandasai import SmartDatalake
from pandasai import Agent
from pandasai.llm import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from vector_search import MetaEngine
from utils import load_templates, get_template, WordContext

logging.basicConfig(filename='prebuilt.log', format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)


class Extractor:

    def __init__(self,
                 openai_api_key: str,
                 customer_request: str,
                 make_plot: bool = False,
                 selected_tables: List[str] = None):

        os.environ['OPENAI_API_KEY'] = openai_api_key

        self.openai_api_key = openai_api_key
        self.customer_request = customer_request
        self.make_plot = make_plot
        self.llm = OpenAI(api_token=openai_api_key, model="gpt-4", max_tokens=1000)

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
        self.response = None

        self.selected_table_keys = []
        self.selected_tables = []

        self.dl = None

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
        """agent = Agent(self.selected_tables, config={"llm": self.llm}, memory_size=10)
        # Chat with the agent
        response = agent.chat(self.customer_request)
        print(response)
        # Get Clarification Questions
        questions = agent.clarification_questions()
        for question in questions:
            print(question)
        response = agent.explain()
        print(response)"""

        # vals =  "number", "dataframe", "plot", "string"

        # use uuid for each request
        # for each csv and png print result
        # png convert to base 64

        """
        import base64

        with open("grayimage.png", "rb") as img_file:
         b64_string = base64.b64encode(img_file.read())
        sys.stdout.write(f"TYPE: {type}")
        sys.stdout.write(b64_string)"""

        self.clean_out_dirs('output_tables', "*.csv")
        self.clean_out_dirs('output_plot', "*.png")

        self.dl = SmartDatalake(self.selected_tables,
                                config={"save_charts": True,
                                        "save_charts_path": "./output_plot",
                                        "llm": self.llm})

        self.response = self.dl.chat(self.customer_request, output_type="dataframe")

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        self.response.to_csv(f'./output_tables/table_{time_stamp}.csv')

        if self.make_plot:
            self.response.chat("Create a plot of your result.")

    def add_request(self, request):
        # self.clean_out_dirs('output_tables', "*.csv")
        # self.clean_out_dirs('output_plot', "*.png")

        new_answer = self.response.chat(request)
        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        new_answer.to_csv(f'./output_tables/table_{time_stamp}.csv')

    def new_request(self, tables, request):
        self.dl = SmartDatalake(tables,
                                config={"save_charts": True,
                                        "save_charts_path": "./output_plot",
                                        "llm": self.llm})

        new_response = self.dl.chat(request, output_type="dataframe")

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)
        new_response.to_csv(f'./output_tables/table2_{time_stamp}.csv')

    @staticmethod
    def clean_out_dirs(dir_path, file_type):
        filelist = glob.glob(os.path.join(dir_path, file_type))
        for f in filelist:
            os.remove(f)

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
