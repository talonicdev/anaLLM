from enum import Enum

import numpy as np
from dateutil.parser import parse

import yaml
import openpyxl
import re
import os
import openai
import pandas as pd

from pathlib import Path
from decouple import config

from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate


os.environ['API_USER'] = config('USER')
os.environ['OPENAI_API_KEY'] = config('KEY')
os.environ['API_KEY'] = config('API_KEY')


class ExtendedEnum(Enum):
    @classmethod
    def abbreviations(cls):
        return list(map(lambda c: re.findall(r'\(.*?\)', c.value)[0][1:-1].lower(), cls))

    @classmethod
    def names(cls):
        return list(map(lambda c: c.value.split(' (')[0].lower(), cls))


class Units(ExtendedEnum):
    pass


class WordContext(Enum):
    pass


class WordIndustries(Enum):
    pass


class WordException(Enum):
    MANAGEMENT = "management"
    ROW = "row"
    COLUMN = "column"
    DATA = "data"
    ANALYSIS = "analysis"
    ANALYTICS = "analytics"


def format_dataframe(file_path: str):
    dataframe = openpyxl.load_workbook(file_path)
    dataframe1 = dataframe.active
    row_vectors = []

    for row in dataframe1.iter_rows(1, dataframe1.max_row):
        row_vectors.append([i for i, cell in enumerate(row) if cell.value])

    row_lens = [len(row) for row in row_vectors]
    max_row_len = np.max(row_lens)
    keep_idx = [i for i in range(len(row_lens)) if row_lens[i] > max_row_len//2]

    data = []
    col_names = []
    side_data = []

    for idx, row in enumerate(dataframe1.iter_rows(1, dataframe1.max_row)):
        if idx in keep_idx:
            if idx == keep_idx[0]:
                col_names = [cell.value for cell in row]
            else:
                vals = {col_names[i]: cell.value for i, cell in enumerate(row)}
                data.append(vals)
        else:
            if idx < keep_idx[0]:
                vals = [cell.value for cell in row if cell.value]
                if len(vals) > 0:
                    side_data.extend([cell.value for cell in row if cell.value])

    df = pd.DataFrame.from_records(data)
    df = df.fillna(value=np.nan)

    if len(side_data) == 0:
        side_data.append(Path(file_path).stem)

    return df, side_data


def load_templates(template_name: str) -> tuple:
    script_path = Path(__file__).parents[1].resolve()

    with open(script_path / f'{template_name}/example_template.txt', 'r') as file:
        template = file.read().replace('\n', ' \n ')

    with open(script_path / f'{template_name}/prefix.txt', 'r') as file:
        prefix = file.read().replace('\n', ' \n ')

    with open(script_path / f'{template_name}/suffix.txt', 'r') as file:
        suffix = file.read()

    with open(script_path / f'{template_name}/examples.yaml', 'r') as file:
        examples = yaml.safe_load(file)

    examples = [examples[k] for k in examples.keys()]
    return template, prefix, suffix, examples


def get_template(examples, prefix, suffix) -> ChatPromptTemplate:
    example_prompt = ChatPromptTemplate.from_messages(
        [('human', suffix), ('ai', '{answer}')]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )
    return ChatPromptTemplate.from_messages([('system', prefix), few_shot_prompt, ('human', suffix)])


def load_datasets(subset: int = -1):
    script_path = Path(__file__).parent.resolve()
    datasets_path = script_path / 'datasets/original'

    file_list = [f for f in datasets_path.glob('**/*') if f.is_file()][:subset]

    table_list = []

    for file in file_list:
        file_extension = file.suffix
        table = pd.read_csv(file) if file_extension == '.csv' else pd.read_excel(file)
        table_list.append(table)

    return table_list


def parse_string(elem: str) -> dict:
    numbers = []
    others = []
    idx = 0

    p = re.compile("([a-zA-Z]+)")
    matches = p.finditer(elem)
    matches = tuple(matches)
    for i, m in enumerate(p.finditer(elem)):
        others.append(m.group())
        if elem[idx: m.start()]:
            numbers.append(elem[idx: m.start()])
        if i == len(matches) - 1:
            if elem[m.end():]:
                numbers.append(elem[m.end():])
        idx = (m.start() + len(m.group()))

    return {'numbers_list': numbers, 'others_list': others}


def parse_number(numbers_list: list, others_list: list):
    separators = ['.', ',']
    seps = []
    unit = None
    for e in others_list:
        if e in separators:
            seps.append(e)
        else:
            unit = e

    if len(seps) > 1:
        float_digit = float(''.join(numbers_list[: -1])) + float('0.' + numbers_list[-1])
    elif len(seps) == 1:
        if seps[0] == ',':
            float_digit = float('.'.join(numbers_list))
        else:
            float_digit = float('.'.join(numbers_list))
    else:
        float_digit = int(numbers_list[0])

    return float_digit, unit


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def is_range(string):
    if '-' in string:
        if len(string.split('-')) == 2:
            start, end = string.split('-')
            if len(start) > 0:
                if start.isnumeric() and end.isnumeric():
                    return True
    else:
        return False


def special_char(elem: str):
    special = re.compile("[!@#&*()_+=|<>?{}\\[\\]~]")
    return special.search(elem)


def check_number_unit(elem: str, units):
    res = parse_string(elem)
    if len(res['others_list']) > 0:
        if np.all([x.lower() in units.names() + units.abbreviations() for x in res['others_list']]):
            return True
    else:
        return False


def prep_table(table: pd.DataFrame):
    table = table.dropna()
    table = table.replace([np.inf, -np.inf], 0)
    x = table.infer_objects().dtypes
    for i, col in enumerate(table.columns):
        if x[col] == 'object':
            if np.all([isinstance(x, str) for x in table[col].tolist()]):
                if np.all([bool(re.search(r'\d', x)) for x in table[col].tolist()]):
                    if not np.any([is_date(x) for x in table[col].tolist()]):
                        if not np.any([is_range(x) for x in table[col].tolist()]):
                            if not np.any([special_char(x) for x in table[col].tolist()]):
                                if len(parse_string(table[col].iloc[0])['others_list']) < 3:
                                    if check_number_unit(table[col].iloc[0], Units):
                                        val_units_df = table[col].apply(lambda x: parse_number(**parse_string(x)))
                                        df_new = pd.DataFrame.from_records(val_units_df.to_list(),
                                                                           columns=[f'{col}', 'unit'])
                                        table.drop(columns=col, inplace=True)
                                        table = pd.concat([table, df_new], axis=1)

    return table
