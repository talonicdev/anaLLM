import argparse
import os
import glob
import logging
import sys

from pathlib import Path

import pandas as pd
from decouple import config

from prebuilt import Extractor
from request_engine import TableSetter

logging.basicConfig(filename='./logs/prebuilt.log', format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)

logging.info("Meta data prediction:")


class PredictMeta:

    meta_data = [
        "Industry",
        "Markets",
        "Business Model",
        "Product Categories",
        "Operating regions",
        "Sourcing regions",
        "Distribution Channels",
        "Employee count",
        "Customer count"
    ]

    def __init__(self,
                 openai_api_key: str,
                 tables: str):

        self.openai_api_key = openai_api_key
        self.tables_dir = Path(tables_dirs)
        self.tables = []

    def load_tables(self):
        """
        Loading all tables from this company.
        :return:
        """

        for f_path in list(self.tables_dir.iterdir()):
            file_extension = f_path.suffix

            if f_path == Path("/Users/stella/talonic/anaLLM/datasets/original/Netflix Userbase.csv"):

                if file_extension in ['.csv', '.xlsx']:
                    self.tables.append(pd.read_csv(f_path)) if file_extension == '.csv' else self.tables.append(pd.read_excel(
                        f_path))

                else:
                    self.tables.append(pd.read_json(f_path))

    def get_meta(self):
        for key in self.meta_data:
            self.get_data(key)

    def get_data(self, meta):
        extraction = Extractor(os.environ['KEY'],
                               "None", make_plot=False)

        extraction.new_request(self.tables, f"What is the {meta} of this dataframe?", "string")
        sys.stdout.write(extraction.new_response)
        extraction.clean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-o', '--openai_api_key', help='OPENAI Key', required=True)
    parser.add_argument('-d', '-table_dirs', help='irectory with all tables.', required=True)
    args = parser.parse_args()

    m = PredictMeta(**vars(args))
    m.load_tables()
    m.get_meta()





