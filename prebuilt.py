import ast
import os
import logging
import argparse
import sys
from typing import List
import glob, os.path
import requests

from aenum import extend_enum

import pandas as pd
import yaml
import pandasai

from pathlib import Path
from pandasai import SmartDatalake, Agent
from pandasai.llm import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from vector_search import MetaEngine
from utils.conf import load_templates, get_template, WordContext

logging.basicConfig(filename='prebuilt.log', format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)


class Extractor:

    def __init__(self,
                 openai_api_key: str,
                 api_key: str,
                 token: str,
                 customer_request: str,
                 make_plot: bool = False,
                 selected_tables: List[str] = None,
                 debug=False):

        """
        Initialize a request object.
        :param openai_api_key: openai api key
        :param customer_request: request from customer
        :param make_plot: options: True / False - if for this specific request a plot be generated
        :param selected_tables: use only specific tables to obtain the answer to your request
        """

        os.environ['OPENAI_API_KEY'] = openai_api_key

        self.api_key = api_key
        self.token = token
        self.debug = debug
        self.openai_api_key = openai_api_key
        self.customer_request = customer_request
        self.make_plot = make_plot
        self.llm = OpenAI(api_token=openai_api_key, model="gpt-4", max_tokens=1000) #
        self.cwd = Path(__file__).parent.resolve()

        self.load_meta_table()

        if selected_tables:
            self.get_selected_tables(selected_tables)
        self.load_WordContext()

        self.prompt_template = None
        self.meta_data_table = None

        self.template = None
        self.examples = None
        self.prefix = None
        self.suffix = None

        self.keys_words = None
        self.response = None
        self.new_response = None

        self.selected_table_keys = []
        self.selected_tables = []

        self.dl = None

        # save file name for each table

    def call_table(self, sheet_id):
        base_url = 'https://backend.vhi.ai/service-api'
        headers = {'Authorization': f'Bearer {self.token}',
                   'x-api-key': f'{self.api_key}'}

        response = requests.get(f"{base_url}/sheet/{sheet_id}", headers=headers)
        if response.status_code == 200:
            sheet_data = response.json()
            return pd.DataFrame(sheet_data['sheet'])
        else:
            print("Error:", response.status_code, response.text)

    def load_meta_table(self):
        base_url = 'https://backend.vhi.ai/service-api'
        headers = {'Authorization': f'Bearer {self.token}',
                   'x-api-key': f'{self.api_key}'}

        result = requests.get(f"{base_url}/metadata", headers=headers)

        meta = result.content
        meta = meta.decode('utf-8')
        import json
        meta_dict = json.loads(meta)
        del meta_dict['userId']
        del meta_dict['column_type']
        del meta_dict['scopes']
        self.meta_data_table = pd.DataFrame(meta_dict)

    def get_selected_tables(self,
                            selected_tables):
        """
        load selected tables
        :param selected_tables: list of paths
        """
        import copy
        selected_df = list()
        for name in selected_tables:
            table = self.meta_data_table.loc[self.meta_data_table['table_name'] == name]
            selected_df.append(copy.deepcopy(table))

        self.meta_data_table = pd.concat(selected_df)
        self.meta_data_table.reset_index(drop=True, inplace=True)

    def get_meta_template(self):
        """
        load the meta data template
        """
        self.template, self.prefix, self.suffix, self.examples = load_templates('meta_template')
        self.prompt_template = get_template(self.examples,
                                            self.prefix,
                                            self.suffix)

    def key_word_selection(self):
        """
        It selects useful key words from the given request
        """
        prompt = self.prompt_template.format(question=self.customer_request)
        prompt_template = ChatPromptTemplate.from_template(prompt)
        message = prompt_template.format_messages()

        llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv(self.openai_api_key), model_name="gpt-4-1106-preview")
        response = llm(message)

        self.keys_words = ast.literal_eval(response.content)

    def select_tables(self):
        """
        It selects useful tables indices to answer the given request based on key words.
        Results are in selected_table_keys.
        """

        me = MetaEngine(self.token)
        me.load_collection('talonic_collection')  # sheets for all users
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

    def get_tables(self):
        """
        Loads the tables based on the selected tables indices. Results are in selected_tables.
        """
        logging.info(f"SELECTED TABLES")
        pd.set_option('display.max_columns', None)

        for key in self.selected_table_keys:
            table = self.call_table(key)
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

        self.dl = Agent(self.selected_tables,
                        config={"save_charts": True,
                                "save_charts_path": f"{self.cwd}/output_plot",
                                "llm": self.llm},
                        memory_size=10)
        self.response = self.dl.chat(self.customer_request)
        explanation = self.dl.explain()

        if isinstance(self.response, str):
            sys.stdout.write(f"TYPE: string\n")
            sys.stdout.write(self.response)
            sys.stdout.write("\n")

            # add explain
            sys.stdout.write(f"EXPLAIN: string\n")
            sys.stdout.write(explanation)
            sys.stdout.write("\n")

        elif isinstance(self.response, dict):
            # TODO: need an example with dict output
            pass

        elif isinstance(self.response, pandasai.smart_dataframe.SmartDataframe):
            self.response.to_csv(f'{self.cwd}/output_tables/table_{self.get_uuid_name()}.csv')

        if self.make_plot:
            self.response.chat("Create a plot of your result.")

        self.clean()

    def clean(self):
        """
        Cleans each temp directory.
        """
        self.write_out_files()

        self.clean_out_dirs(f'{self.cwd}/output_tables', "*.csv")
        self.clean_out_dirs(f'{self.cwd}/output_plot', "*.png")

    @staticmethod
    def write_out_files():
        """
        Writes out the results of a request.
        """
        import base64
        cwd = Path(__file__).parent.resolve()

        filelist = glob.glob(os.path.join(f'{cwd}/output_plot', '*.png'))
        for f in filelist:
            with open(f, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                sys.stdout.write(f"TYPE: image\n")
                sys.stdout.write(encoded_string)
                sys.stdout.write("\n")

        filelist = glob.glob(os.path.join(f'{cwd}/output_tables', '*.csv'))
        for f in filelist:
            sys.stdout.write(f"TYPE: table\n")
            with open(f, "r") as my_input_file:
                for idx, line in enumerate(my_input_file):
                    line = line.split(",")[1:]
                    line[-1] = line[-1][:-1]
                    sys.stdout.write(",".join(line))
            sys.stdout.write("\n")

    @staticmethod
    def get_uuid_name():
        """
        Get uuid.
        """
        import uuid
        return uuid.uuid1()

    def add_request(self,
                    request: str):
        """
        Add to existing a new request.
        :param request: request based on previous one.
        """
        new_answer = self.response.chat(request)
        new_answer.to_csv(f'{self.cwd}/output_tables/table_{self.get_uuid_name()}.csv')

    def new_request(self, tables, request, output_type):
        """
        Create a new request within the same session.
        :param tables: table to be used.
        :param request: customer request
        :param output_type: type of output. ex.: dataframe, str
        """
        self.dl = SmartDatalake(tables,
                                config={"save_charts": True,
                                        "save_charts_path": "./output_plot",
                                        "llm": self.llm})

        self.new_response = self.dl.chat(request, output_type=output_type)

        if output_type == "dataframe":
            self.new_response.to_csv(self.cwd / f'output_tables/table_new_{self.get_uuid_name()}.csv')

    @staticmethod
    def clean_out_dirs(dir_path, file_type):
        """
        Cleans a specific temp directory.
        """
        filelist = glob.glob(os.path.join(dir_path, file_type))
        for f in filelist:
            os.remove(f)

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

    @staticmethod
    def load_WordIndustries():
        """
        Loads context attributes into enum class named WordContext.
        """
        if Path('utils/WordIndustries.yaml').is_file():
            with open('utils/WordIndustries.yaml', 'r') as f:
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
    extraction.run_request()
    extraction.clean()
