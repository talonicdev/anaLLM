import os
import glob
import logging
import requests

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
os.environ['API_KEY'] = config('API_KEY')


def test_complete_table():
    # 1. https://appdev.talonic.ai laden
    # 2. develop in safari öffnen
    # 3. show javascript console
    # 4. enter window.sessionStorage.getItem('accessToken')
    # 5. copy token

    '''api_key = os.environ['API_KEY']
    token = "b497e715-0373-4f48-a531-96a1d6fe8af3"
    base_url = 'https://backend.vhi.ai/service-api'
    headers = {'Authorization': f'Bearer {token}',
               'x-api-key': f'{api_key}'}

    response = requests.get(f"{base_url}/sheet-overview", headers=headers)
    if response.status_code == 200:
        all_sheets = response.json()'''

    token = "b497e715-0373-4f48-a531-96a1d6fe8af3"
    sheet_id = "333d2d8e-99e9-4398-ac0b-f97b1b68cf95"
    users_prompt = "what are the preferred payments only for male?"

    ct = CompleteTable(openai_api_key=os.environ['KEY'],
                       table_path='./rating_table.xlsx',
                       token=token,
                       users_prompt=users_prompt,
                       api_key=os.environ['API_KEY'],
                       sheet_id=sheet_id,
                       debug=False)
    e_cols, o_cols = ct.get_empty_cols()
    c_request, useful, exists_cols, empty_cols = ct.get_table_question(e_cols, o_cols)
    ct.create_table(c_request, useful, exists_cols, empty_cols)


def test_table_setter():
    api_key = os.environ['API_KEY']
    token = "b497e715-0373-4f48-a531-96a1d6fe8af3"
    base_url = 'https://backend.vhi.ai/service-api'
    headers = {'Authorization': f'Bearer {token}',
               'x-api-key': f'{api_key}'}

    response = requests.get(f"{base_url}/sheet-overview", headers=headers)
    if response.status_code == 200:
        all_sheets = response.json()
        for i in range(len(all_sheets)):
            sheet_id = all_sheets[i]['sheetId']
            # sheet_id = "eb2eb73d-1d52-45ea-858c-5ba54972f937"
            setter = TableSetter(os.environ['KEY'],
                                 api_key=api_key,
                                 token=token,
                                 sheet_id=sheet_id,
                                 new_collection=False,
                                 debug=False)
            setter.run(destination_name='app_test')


def test_ask():
    # ['products_data', 'Adidas US Sales Datasets']

    logging.info("TESTING Quality of Software:")

    token = "b497e715-0373-4f48-a531-96a1d6fe8af3"

    extraction = Extractor(os.environ['KEY'],
                           token=token,
                           api_key=os.environ['API_KEY'],
                           customer_request="Summarize the sales of Apparel products per city, but exclude all sales "
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
    # test_table_setter()
    test_complete_table()
