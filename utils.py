from enum import Enum
from typing import List

import yaml
import pandas as pd

from pathlib import Path

from langchain import PromptTemplate
from langchain import FewShotPromptTemplate


class WordContext(Enum):
    pass


class WordException(Enum):
    MANAGEMENT = "management"
    ROW = "row"
    COLUMN = "column"
    DATA = "data"
    ANALYSIS = "analysis"
    ANALYTICS = "analytics"


def load_templates(template_name: str) -> tuple:
    script_path = Path(__file__).parent.resolve()

    with open(script_path / f'{template_name}/example_template.txt', 'r') as file:
        template = file.read().replace('\n', ' \n ')

    with open(script_path / f'{template_name}/prefix.txt', 'r') as file:
        prefix = file.read().replace('\n', ' \n ')

    with open(script_path / f'{template_name}/suffix.txt', 'r') as file:
        suffix = file.read().replace('\n', ' \n ')

    with open(script_path / f'{template_name}/examples.yaml', 'r') as file:
        examples = yaml.safe_load(file)

    examples = [examples[k] for k in examples.keys()]
    return template, prefix, suffix, examples


def get_template(template, examples, prefix, suffix, input_variables: List[str]) -> FewShotPromptTemplate:
    example_prompt = PromptTemplate(
        input_variables=input_variables + ["answer"],
        template=template
    )

    prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables,
        example_separator="\n\n"
    )

    return prompt_template


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
