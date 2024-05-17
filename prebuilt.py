import ast
import os
import argparse
import sys
from typing import List
import glob
import os.path
import pprint

from aenum import extend_enum

import pandas as pd
import pandasai.constants
import yaml
import pandasai

from pathlib import Path
from pandasai import SmartDatalake, Agent, SmartDataframe
from pandasai.connectors import BaseConnector, PandasConnector
from langchain.prompts import ChatPromptTemplate

from vector_search import VectorSearch
from utils.conf import load_templates, get_template, WordContext, format_dataframe, prep_table

from common import Common, Requests, WriteType, ProcessErrType, get_logger
from config import Config

from typing import Tuple

logger = get_logger(__name__)

# Overwrite PandasAI's whitelisted builtins 
# and add exceptions to imports and libraries
pandasai.constants.WHITELISTED_BUILTINS.extend([
    "Exception","ValueError","NameError","TypeError",
    "ArithmeticError","ZeroDivisionError","KeyError","FloatingPointError",
    "AttributeError","OverflowError","StopIteration"
])
pandasai.constants.WHITELISTED_LIBRARIES.extend(['_strptime'])

class Extractor:

    def __init__(self,
                 openai_api_key: str,
                 api_key: str,
                 token: str,
                 customer_request: str,
                 make_plot: bool = False,
                 selected_tables: List[str] = None,
                 response_type: str = 'string',
                 sheet_id: str = None,
                 debug=False):

        """
        Initialize a request object.
        :param openai_api_key: openai api key
        :param customer_request: request from customer
        :param make_plot: options: True / False - if for this specific request a plot be generated
        :param selected_tables: use only specific tables to obtain the answer to your request
        :param response_type: request a specific kind of response (dataframe, string, number, plot)
        :param sheet_id: use only one specific table to obtain the answer to your request
        """

        os.environ['OPENAI_API_KEY'] = openai_api_key

        self.api_key = api_key
        self.token = token
        self.debug = debug
        self.openai_api_key = openai_api_key
        self.customer_request = customer_request
        self.sheet_id = sheet_id if sheet_id else None
        self.make_plot = make_plot
        self.cwd = Path(__file__).parent.resolve()
        self.response_type = response_type if response_type else "dataframe"
        
        # Initialize config first..
        self.config = Config(
            openai_api_key=openai_api_key,
            token=token,
            api_key=api_key
        )

        self.common = Common(config=self.config)
        self.requests = Requests(config=self.config)
        self.llm = self.common.get_openAI_llm()

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
        self.dl = None

        self.selected_tables = []

        self.init_data()

    def init_data(self):
        if self.debug:
            cwd = Path(__file__).resolve().parent
            if os.path.isfile(f'{cwd}/test_files/test_data/quality_data/quality_meta.xlsx'):
                self.meta_data_table = pd.read_excel(f'{cwd}/test_files/test_data/quality_data/quality_meta.xlsx',
                                             index_col=None)
        else:
            self.load_meta_table()

    def call_table(self, sheet_id, meta) -> Tuple[PandasConnector,str]:
        try:
            response = self.requests.get(f'sheet/{sheet_id}')
            if response.status_code == 200:
                sheet_data = response.json()
                name = f"{sheet_data['tableName']}, {sheet_data['sheetName']}"
                description = ""
                if meta is not None:
                    try:
                        description = meta['description'].values[0]
                    except:
                        pass
                df = pd.DataFrame(sheet_data['sheet'])
                return PandasConnector({"original_df": df}, name=name, description=description), name
                #return pd.read_json(sheet_data['sheet'])
            else:
                print(response)
                self.common.write(WriteType.ERROR, response)
                return None
        except Exception as e:
            self.common.write(WriteType.ERROR, f"Could not call sheet {sheet_id}",error=e)
            return None

    def load_meta_table(self):
        result = self.requests.get('metadata')
        meta = result.content
        meta = meta.decode('utf-8')
        import json
        meta_dict = json.loads(meta)
        del meta_dict['userId']
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
        llm = self.common.get_chatOpenAI_llm(temperature=0)
        response = self.common.invoke_chatOpenAI(llm, message)

        response_eval = ast.literal_eval(response.content)
        self.keys_words = [val for val in response_eval if val.strip()]
        self.common.write(WriteType.DEBUG, f'Keywords: {self.keys_words}')

    def select_tables(self):
        """
        It selects useful tables indices to answer the given request based on key words.
        Results are in selected_table_keys.
        """
        name = 'test_collection' if self.debug else self.config.COLLECTION_NAME
        vs = VectorSearch(self.token, debug=self.debug, collection_name=name, sheet_id=self.sheet_id)
        table_keys = vs.find_tables(self.customer_request)
        self.get_tables(table_keys)

    def get_tables(self, table_keys):
        """
        Loads the tables based on the selected tables indices. Results are in selected_tables.
        """
        #logging.info(f"SELECTED TABLES")
        # cwd for table path
        retrieved_keys = []
        dfs = []
        
        for key in table_keys:
            if self.debug:
                cwd = Path(__file__).resolve().parent
                table_path = self.meta_data_table[self.meta_data_table['key'] == key]['path'].tolist()[0]
                p = f'{cwd}/test_files/test_data/quality_data/{table_path}'
                table, table_side_data = format_dataframe(p)
                # table = prep_table(table)
                self.selected_tables.append(table)

            else:
                meta = self.meta_data_table[self.meta_data_table['key'] == key]
                table, name = self.call_table(key, meta)
                if table is not None:
                    self.selected_tables.append(table)
                    retrieved_keys.append(key)
                    dfs.append(name)
                else:
                    self.common.write(WriteType.DEBUG, f'Selected sheet "{key}" could not be retrieved')

        self.common.write(WriteType.DATAFRAMES, dfs)
        #self.common.write(WriteType.REFERENCES, retrieved_keys)
        
    def try_run(self, prompt: str, n_attempts_left: int = 0) -> bool:
        if n_attempts_left > 0:
            self.dl.clear_memory()
            self.response = self.dl.chat(prompt, self.response_type)
            response_type, is_valid_type = self.common.get_response_type(self.response_type,self.response)
            if is_valid_type:
                self.response_type = response_type
                return True
            else:
                n_attempts_left = n_attempts_left - 1
                return self.try_run(prompt,n_attempts_left=n_attempts_left)
        else:
            self.response_type = "error"
            return False

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
        
        if not len(self.selected_tables):
            # No sheets to load -> No Agent to initialize -> Fail gracefully due to no matching sheets to work with
            self.common.write(WriteType.PROCESSERR,ProcessErrType.NO_MATCHING_SHEETS)
            raise SystemExit(0)

        self.dl = Agent(self.selected_tables,
                        config={"save_charts": True,
                                "save_charts_path": f"{self.cwd}/output_plot",
                                "llm": self.llm,
                                "save_logs": self.debug,
                                "enable_cache": False,
                                "verbose": self.debug,
                                "llm_options": {
                                    "request_timeout": self.config.REQUEST_TIMEOUT
                                },
                                "max_retries": self.config.ERROR_CORRECTION_ATTEMPTS
                                },
                        memory_size=10)
        
        #self.response = self.common.chat_agent(self.dl,self.customer_request,self.response_type)
        prompt = f"{self.customer_request}\n**Important:** Always expect invalid data types or missing values for individual cells and catch exceptions and/or check with 'isinstance()'."
        if self.response_type == "dataframe":
            prompt = prompt + f"\nPlease also ensure that all new values in the resulting dataframe are simple types (string, number or boolean) as we want to save the table in a CSV file later."
        
        result = self.try_run(prompt,self.config.MAX_RETRIES)
        
        self.common.write(WriteType.CODE,self.dl.last_code_generated)
        
        if result:
            self.common.write(WriteType.RESULT,{'type':self.response_type,'data':self.response})
        else:
            res_type = type(self.response).__name__
            self.common.write(WriteType.WARN,f'Expected type {self.response_type}, got {res_type}')
            self.common.write(WriteType.ERROR,self.dl.last_error)
            self.common.write(WriteType.PANDASAIERR,self.dl.last_error)
        self.response = self.dl.chat(prompt, self.response_type)

        #if self.make_plot:
        #    self.response.chat("Create a plot of your result.")

        #if self.config.LOG_LEVEL == 'trace':
        #    sys.stdout.write(f"TYPE: LOGS\n: {pprint.pformat(self.dl._lake.logs)}")
            
        # Uncomment to print the full SmartDataLake logs
        #sys.stdout.write(f"TYPE: LOGS\n: {pprint.pformat(self.dl._lake.logs)}")

        #self.clean()

    def clean(self):
        """
        Cleans each temp directory.
        """
        self.write_out_files()

        #self.clean_out_dirs(f'{self.cwd}/output_tables', "*.csv")
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
                sys.stdout.write(f"TYPE: IMAGE\n")
                sys.stdout.write(encoded_string)
                sys.stdout.write("\n")

        filelist = glob.glob(os.path.join(f'{cwd}/output_tables', '*.csv'))
        for f in filelist:
            sys.stdout.write(f"TYPE: TABLE\n")
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
                                        "enable_cache": False,
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
    parser = argparse.ArgumentParser(description='Extractor module')
    parser.add_argument('-o', '--openai_api_key', help='OPENAI Key', required=True)
    parser.add_argument('-a', '--api_key', help='Backend Key', required=True)
    parser.add_argument('-token', '--token', required=True)
    parser.add_argument('-r', '--customer_request', help='Task for AI.', required=True)
    parser.add_argument('-sheet', '--sheet_id')
    parser.add_argument('-response', '--response_type')
    parser.add_argument('--selected_tables', help='A list of table names that should be selected.')
    args = parser.parse_args()

    extraction = Extractor(**vars(args))
    extraction.get_meta_template()
    extraction.key_word_selection()
    extraction.select_tables()
    extraction.run_request()
    #extraction.clean()
