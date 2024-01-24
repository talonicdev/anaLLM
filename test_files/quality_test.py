import os
import glob
import logging

import pandas as pd

from pathlib import Path
from decouple import config

from complete import CompleteTable
from prebuilt import Extractor
from request_engine import TableSetter

logging.basicConfig(filename='../prebuilt.log', format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)

os.environ['API_USER'] = config('USER')
os.environ['OPENAI_API_KEY'] = config('KEY')


def test_complete_table():
    '''file = open('./test_files/test_file.txt', "r")
    content = file.read()
    ct = CompleteTable(openai_api_key=os.environ['KEY'], content=content)'''

    '''df = pd.read_excel('/Users/stella/talonic/anaLLM/test_files/multi_table.xlsx')
    df.columns = df.iloc[0].to_list()
    df = df.iloc[1:]
    df.drop(columns=['Expense per Unit'], inplace=True)
    df.to_excel('/Users/stella/talonic/anaLLM/test_files/single_table.xlsx')'''

    # ct = CompleteTable(openai_api_key=os.environ['KEY'], table_path='./multi_table.xlsx')
    ct = CompleteTable(openai_api_key=os.environ['KEY'], table_path='./rating_table.xlsx')
    e_cols, o_cols = ct.get_empty_cols()
    c_request, useful, exists_cols, empty_cols = ct.get_table_question(e_cols, o_cols)
    ct.create_table(c_request, useful, exists_cols, empty_cols)


def insert():
    script_path = Path(__file__).parent.resolve()
    files_path = glob.glob(f"{str(script_path)}/datasets/original/*")

    files = {Path(p).stem: Path(p).name for p in files_path}

    for n in files.keys():
        setter = TableSetter(os.environ['KEY'], files[n])
        setter.run(destination_name=f'{n}')


def ask():
    # ['products_data', 'Adidas US Sales Datasets']

    logging.info("TESTING Quality of Software:")

    extraction = Extractor(os.environ['KEY'],
                           "Summarize the sales of Apparel products per city, but exclude all sales "
                           "that had an operating margin of less than 30%.", make_plot=True)
    extraction.get_meta_template()
    extraction.key_word_selection()
    extraction.select_tables()
    extraction.run_request()
    extraction.clean()
    # extraction.add_request('And how many are less than 15% of them?')

    requests = [
        "Summarize the sales of Apparel products per city, but exclude all sales that had an operating margin of less than 30%.",
        "Show the performance of the different ad spends in relation to the gender and sales.",
        "Compare the operating margins between the sales methods and list their differences per city.",
        "What the best distribution of budget between TV, Radio and Newspaper Ads is and list their ROAS?",
        "Check if there is a correlation between pay zones and the reason why people left.",
        "Give me some insights about the relation of subscription types, location, genders and devices.",
        "Are there any patterns in the job roles and the duration of employment?",
        "Analyze whether the employee performance has a relation to the frequency of termination."
    ]

    '''for customer_request in requests:

        logging.info(f"customer request: {customer_request}")

        extraction = Extractor(api_key, customer_request)
        extraction.get_meta_template()
        extraction.key_word_selection()
        extraction.select_tables()'''


if __name__ == "__main__":
    test_complete_table()
