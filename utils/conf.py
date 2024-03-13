from enum import Enum
from dateutil.parser import parse

import yaml
import pandas as pd

from pathlib import Path

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate


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
    elem = elem.replace(" ", "")
    sub_str = ''

    if elem[0].isdigit():
        current_elem_type = 'numeric'
    else:
        current_elem_type = 'str'

    for i in range(len(elem)):
        if elem[i].isdigit():
            if current_elem_type == 'numeric':
                sub_str += elem[i]
                if len(elem) == i + 1:
                    numbers.append(sub_str)
            else:
                others.append(sub_str)
                sub_str = elem[i]
                current_elem_type = 'numeric'
                if len(elem) == i + 1:
                    numbers.append(sub_str)

        else:
            if current_elem_type == 'str':
                sub_str += elem[i]
                if len(elem) == i + 1:
                    others.append(sub_str)
            else:
                numbers.append(sub_str)
                sub_str = elem[i]
                current_elem_type = 'str'
                if len(elem) == i + 1:
                    others.append(sub_str)

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
        start, end = string.split('-')
        if len(start) > 0:
            return True
    else:
        return False

