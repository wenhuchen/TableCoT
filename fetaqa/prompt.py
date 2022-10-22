import os
import openai
import json
import random
import argparse
import tqdm
import sys
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--start", required=True, type=int)
parser.add_argument("--end", required=True, type=int)
parser.add_argument("--dry_run", default=False, action="store_true",
    help="whether it's a dry run or real run.")
parser.add_argument(
    "--temperature", type=float, default=0.7,
    help="temperature of 0 implies greedy sampling.")


demonstration = """
Read the following table regarding "Shagun Sharma" to answer the given question.

Year | Title | Role | Channel
2015 | Kuch Toh Hai Tere Mere Darmiyaan | Sanjana Kapoor | Star Plus
2016 | Kuch Rang Pyar Ke Aise Bhi | Khushi | Sony TV
2016 | Gangaa | Aashi Jhaa | &TV
2017 | Iss Pyaar Ko Kya Naam Doon 3 | Meghna Narayan Vashishth | Star Plus
2017–18 | Tu Aashiqui | Richa Dhanrajgir | Colors TV
2019 | Laal Ishq | Pernia | &TV
2019 | Vikram Betaal Ki Rahasya Gatha | Rukmani/Kashi | &TV
2019 | Shaadi Ke Siyape | Dua | &TV

Question: What TV shows was Shagun Sharma seen in 2019?
Answer: In 2019, Shagun Sharma played in the roles as Pernia in Laal Ishq, Vikram Betaal Ki Rahasya Gatha as Rukmani/Kashi and Shaadi Ke Siyape as Dua.
"""

if __name__ == "__main__":
    args = parser.parse_args()

    openai.api_key = os.getenv('OPENAI_KEY')

    with open(f'test_qa.json') as f:
        fetaqa = json.load(f)

    now = datetime.now() 
    dt_string = now.strftime("%d_%H_%M")

    keys = list(fetaqa.keys())[args.start:args.end]

    fw = open(f'outputs/response_s{args.start}_e{args.end}_{dt_string}.json', 'w')
    tmp = {'demonstration': demonstration}
    fw.write(json.dumps(tmp) + '\n')

    for key in tqdm.tqdm(keys):
        entry = fetaqa[key]

        question = entry['question']
        answer = entry['answer']

        prompt = demonstration + '\n'
        prompt += f'Read the following table regarding {entry["title"]} to answer the given question.'
        prompt += entry['table'] + '\n\n'
        prompt += 'Question: ' + question + '\nAnswer:'

        if args.dry_run:
            print(prompt)
        else:
            response = openai.Completion.create(
              model="text-davinci-002",
              prompt=prompt,
              temperature=0.7,
              max_tokens=64,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
            )

            response = response['choices'][0]["text"].strip().strip('\n')

            tmp = {'key': key, 'question': question, 'response': response, 'answer': answer, 'table_id': entry['table_id']}

            fw.write(json.dumps(tmp) + '\n')

    fw.close()
