import os
import logging

from decouple import config
from fastapi import FastAPI
from pydantic import BaseModel

from complete import CompleteTable
from meta_data_prediction import PredictMeta
from prebuilt import Extractor
from request_engine import TableSetter

logging.basicConfig(filename='../endpoints.log', format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)

os.environ['API_USER'] = config('USER')
os.environ['OPENAI_API_KEY'] = config('KEY')
os.environ['API_KEY'] = config('API_KEY')


app = FastAPI()


class GeneralVars(BaseModel):
    token: str
    users_prompt: str | None = None
    customer_request: str | None = None


@app.put("/complete-table/{sheet_id}")
async def run_complete_table(var: GeneralVars, sheet_id: str, debug: bool | None = False):
    ct = CompleteTable(openai_api_key=os.environ['KEY'],
                       token=var.token,
                       users_prompt=var.users_prompt,
                       api_key=os.environ['API_KEY'],
                       sheet_id=sheet_id,
                       debug=debug)
    e_cols, o_cols = ct.get_empty_cols()
    c_request, useful, exists_cols, empty_cols = ct.get_table_question(e_cols, o_cols)
    ct.create_table(c_request, useful, exists_cols, empty_cols)


@app.put("/table-setter/{sheet_id}")
async def run_table_setter(var: GeneralVars, sheet_id: str,  debug: bool | None = False):
    setter = TableSetter(os.environ['KEY'],
                         api_key=os.environ['API_KEY'],
                         token=var.token,
                         sheet_id=sheet_id,
                         new_collection=False,
                         debug=debug)
    setter.run()


@app.post("/extractor")
async def run_extractor(var: GeneralVars):
    extraction = Extractor(os.environ['KEY'],
                           token=var.token,
                           api_key=os.environ['API_KEY'],
                           customer_request=var.customer_request,
                           make_plot=True)

    extraction.get_meta_template()
    extraction.key_word_selection()
    extraction.select_tables()
    extraction.run_request()
    extraction.clean()


@app.get("/table-setter/{sheet_id}")
async def run_table_setter(sheet_id: str):
    m = PredictMeta(openai_api_key=os.environ['API_KEY'],
                    tables=sheet_id)
    m.load_tables()
    m.get_meta()
