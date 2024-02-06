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
os.environ['API_KEY'] = config('API_KEY')


def test_complete_table():
    # 1. https://appdev.talonic.ai laden
    # 2. develop in safari öffnen
    # 3. show javascript console
    # 4. enter window.sessionStorage.getItem('accessToken')
    # 5. copy token

    token = "eyJraWQiOiJQMDhQNXF1MWdQVXhxVkhQODlIR3l1c0JiMWhrcFJJdmhMRENpcGk0N1kwPSIsImFsZyI6IlJTMjU2In0.eyJzdWIiOiJiNDk3ZTcxNS0wMzczLTRmNDgtYTUzMS05NmExZDZmZThhZjMiLCJldmVudF9pZCI6IjU5NzdjN2I5LTM2ZGItNDc5ZS04OGRiLWRmMzNiZWY1MzI1NyIsInRva2VuX3VzZSI6ImFjY2VzcyIsInNjb3BlIjoiYXdzLmNvZ25pdG8uc2lnbmluLnVzZXIuYWRtaW4iLCJhdXRoX3RpbWUiOjE3MDY4ODE5MTUsImlzcyI6Imh0dHBzOlwvXC9jb2duaXRvLWlkcC5ldS1ub3J0aC0xLmFtYXpvbmF3cy5jb21cL2V1LW5vcnRoLTFfYldqenF6amttIiwiZXhwIjoxNzA2ODg1NTE1LCJpYXQiOjE3MDY4ODE5MTUsImp0aSI6Ijc2MmVlYTRmLTlmMjEtNDNlMC05MjQ0LTFiZTQzNDM0ZTgwOCIsImNsaWVudF9pZCI6IjJub2JhODliOXJibjJubDBxZ25ibm9rMGJxIiwidXNlcm5hbWUiOiJiNDk3ZTcxNS0wMzczLTRmNDgtYTUzMS05NmExZDZmZThhZjMifQ.X1Hbcv5EL_MKhBmqxl76ApjEegx8lzmW2m3t07N0lhoLZ3jfIJq3SDuoVDRlrCzvQWUkREh6-eplXeogQafhfEF-dXzLY_PwBWJI00IpHIqqudZaF6QJkKpndLJPnLtbUQPoGEDlKryy-ktyAWYmMZcfsSgTpR41v8xJdv2lK7z_sUeZlYcbnyhfA7DSdckwRpOxY3f-Kwr5OPTbdWbh2jo3Mz-CRqVSZIWnQrkxerhIDJ2SbPyFw2lHUIr-mZ8Xj1X2iV64hl8VVQFILWlNREU8pvysPc2DQk51MCOOHu-bwC-SiuTpslzCEK0LDRUcSJrT1jXCS53BkpUqb1H5_g"
    sheet_id = "eb2eb73d-1d52-45ea-858c-5ba54972f937"

    ct = CompleteTable(openai_api_key=os.environ['KEY'],
                       table_path='./rating_table.xlsx',
                       token=token,
                       api_key=os.environ['API_KEY'],
                       sheet_id=sheet_id,
                       debug=True)
    e_cols, o_cols = ct.get_empty_cols()
    c_request, useful, exists_cols, empty_cols = ct.get_table_question(e_cols, o_cols)
    ct.create_table(c_request, useful, exists_cols, empty_cols)


def test_table_setter():
    token = "eyJraWQiOiJQMDhQNXF1MWdQVXhxVkhQODlIR3l1c0JiMWhrcFJJdmhMRENpcGk0N1kwPSIsImFsZyI6IlJTMjU2In0.eyJzdWIiOiJiNDk3ZTcxNS0wMzczLTRmNDgtYTUzMS05NmExZDZmZThhZjMiLCJldmVudF9pZCI6IjVlMjhiNGQ0LWNhOTQtNGZmYi05OWVlLTJmMjE2YzRjYjAzNiIsInRva2VuX3VzZSI6ImFjY2VzcyIsInNjb3BlIjoiYXdzLmNvZ25pdG8uc2lnbmluLnVzZXIuYWRtaW4iLCJhdXRoX3RpbWUiOjE3MDcyMjQ1NzMsImlzcyI6Imh0dHBzOlwvXC9jb2duaXRvLWlkcC5ldS1ub3J0aC0xLmFtYXpvbmF3cy5jb21cL2V1LW5vcnRoLTFfYldqenF6amttIiwiZXhwIjoxNzA3MjQxNzM5LCJpYXQiOjE3MDcyMzgxMzksImp0aSI6IjRhZWQzZDQwLTA4ZGYtNDAzNS1hODc4LTBiOGExOTU3NmYzMSIsImNsaWVudF9pZCI6IjJub2JhODliOXJibjJubDBxZ25ibm9rMGJxIiwidXNlcm5hbWUiOiJiNDk3ZTcxNS0wMzczLTRmNDgtYTUzMS05NmExZDZmZThhZjMifQ.CskWxOfPI4LSRIDTLsHR7QWfarcDH3E4cYiqSVqxKvcsQcE4Bz7tkItaeRZ1hknjX5tqshtqfzTKaJQhAFlSN0EoFpkVMm66dIJAZgVYebju1W_Di7KKO0cg8vm_IcQjUTTk245Pw--DM8ypKWDy_W5mBZPxpQOc_n4WNKXgw5iMPjQH9gCJefKDCl3eIL2dRk0hiEoKOsNOTnU7lIXyJpBPGF9dWhqdrEASMXirezG6MyUz-qyJ1-QOwd_0fVtlHeKuym0VdoJV1uhIcB40GANeUYnOYyeYTWhXhXNbOwF46j1Xi6xboFlLqv4HcIyCo7XWXjJwGSX9bmGQZXFJ8g"
    sheet_id = "eb2eb73d-1d52-45ea-858c-5ba54972f937"

    setter = TableSetter(os.environ['KEY'],
                         api_key=os.environ['API_KEY'],
                         token=token,
                         sheet_id=sheet_id,
                         new_collection=False,
                         debug=False)
    setter.run(destination_name='app_test')


def test_ask():
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
    test_table_setter()
