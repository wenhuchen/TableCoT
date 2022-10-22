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

"""
# | Track | Film | BPM | Ref
1 | Star Wars Main Title | Star Wars | 180 | -
2 | The Terminator | The Terminator | 185 | -
3 | Theme From E.T. | E.T. | 180 | -
4 | Prologue | Harry Potter and the Philosopher's Stone | 180 | -
5 | Rhythm & Police (K.O.G G3 Mix) | Bayside Shakedown | 175 | -
6 | 007 | 007 | 175 | -
7 | Feel Good Time | Charlie's Angels: Full Throttle | 175 | -
8 | Men In Black | Men in Black | 175 | -
9 | Ghostbusters | Ghostbusters | 175 | -
10 | The Power Of Love | Back to the Future | 175 | -
11 | Unchained Melody | Ghost | 175 | -
12 | May It Be | The Lord of the Rings | 175 | -
13 | I Don't Wanna Miss A Thing (Planet Lution Mix) | Armageddon | 175 | -
14 | My Heart Will Go On (KCP Remix) | Titanic | 175 | -
15 | Never Ending Story | The NeverEnding Story | 175 | -
16 | The Raiders March | Raiders of the Lost Ark | 175 | -
17 | Main Title | The Matrix Reloaded | 181 | -
18 | Theme From Jaws | Jaws | 184 | -
19 | Also Sprach Zarathustra | 2001: A Space Odyssey | 186 | -
20 | Mission: Impossible Theme | Mission: Impossible | 195 | -

Question: Which track has the lowest bpm, and which has the highest on the Speed SFX series?"
Answer: The Speed SFX's lowest bpm is 175 on tracks #5-16, and the highest bpm is 195 on #20 in the Speed series.

Question: Which track has BPM over 185?
Answer: The Sprach Zarathustra has a bpm of 186 and Mission: Impossible Theme has a bpm of 195, both of them are over 185.
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
