import os
import glob
import logging

from pathlib import Path
from decouple import config

from prebuilt import Extractor
from request_engine import TableSetter

logging.basicConfig(filename='../prebuilt.log', format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)

os.environ['API_USER'] = config('USER')
os.environ['OPENAI_API_KEY'] = config('KEY')


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
    ask()
