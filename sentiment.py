import os

import pandas as pd
import wandb
from decouple import config
from openai import OpenAI



def run_rating(feedback):
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    message = [{"role": "assistant", "content": "I give a rating from 0 till 10 based on the user's feedback of a motorcycle."},
               {"role": "user", "content": feedback}]
    temperature = 0.2
    max_tokens = 256
    frequency_penalty = 0.0

    response = client.chat.completions.create(
        model="gpt-4",
        messages=message,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty
    )

    print(response.choices[0].message.content)
    return response.choices[0].message.content


def test_rating():
    rate_dict = {'feedback': [], 'rating': []}
    feedbacks = ["The clutch lever of this motorcycle isn't working as expected.",
                 "The steering dumper rusted after few months. I thing they sold me used vehicle parts.",
                 "The seat is so soft. Seems very good quality leather.",
                 "The rear wheel seems to spin to fast. I want a refund.",
                 "It is called silencer for a good reason. But the motorcycle doesn't seem to have one!"]

    for feedback in feedbacks:
        rating = run_rating(feedback)
        rate_dict['feedback'].append(feedback)
        rate_dict['rating'].append(rating)

    df = pd.DataFrame(rate_dict)
    df.to_excel('./test_files/feedback_rating.xlsx')


test_rating()

'''wandb.init()
response_text = response.choices[0].message.content
tokens_used = response.usage.total_tokens

prediction_table = wandb.Table(columns=["Prompt", "Response", "Tokens", "Max Tokens", "Frequency Penalty", "Temperature"])
prediction_table.add_data(message, response_text, tokens_used, max_tokens, frequency_penalty, temperature)
wandb.log({'predictions': prediction_table})
wandb.finish()'''

