import os
import openai
import json
import random
import argparse
import tqdm
import sys
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--option", default='cot', type=str)
parser.add_argument("--model", default='text-davinci-002', type=str)
parser.add_argument("--start", required=True, type=int)
parser.add_argument("--end", required=True, type=int)
parser.add_argument("--dry_run", default=False, action="store_true",
	help="whether it's a dry run or real run.")
parser.add_argument(
	"--temperature", type=float, default=0.7,
	help="temperature of 0 implies greedy sampling.")

demonstration = {}
demonstration['direct'] = """
Read the table below regarding "2008 Clásica de San Sebastián" to answer the following questions.

Rank | Cyclist | Team | Time | UCI ProTour Points
1 | Alejandro Valverde (ESP) | Caisse d'Epargne | 5h 29' 10 | 40
2 | Alexandr Kolobnev (RUS) | Team CSC Saxo Bank | s.t. | 30
3 | Davide Rebellin (ITA) | Gerolsteiner | s.t. | 25
4 | Paolo Bettini (ITA) | Quick Step | s.t. | 20
5 | Franco Pellizotti (ITA) | Liquigas | s.t. | 15
6 | Denis Menchov (RUS) | Rabobank | s.t. | 11
7 | Samuel Sánchez (ESP) | Euskaltel-Euskadi | s.t. | 7
8 | Stéphane Goubert (FRA) | Ag2r-La Mondiale | + 2 | 5
9 | Haimar Zubeldia (ESP) | Euskaltel-Euskadi | + 2 | 3
10 | David Moncoutié (FRA) | Cofidis | + 2 | 1

read the question first, and then answer the given question. 

Question: which country had the most cyclists finish within the top 10?
The answer is Italy.

Question: how many players got less than 10 points?
The answer is 4.

Question: how many points does the player from rank 3, rank 4 and rank 5 combine to have? 
The answer is 60.

Question: who spent the most time in the 2008 Clásica de San Sebastián. 
The answer is David Moncoutié.
"""

demonstration['cot'] = """
Read the table below regarding "2008 Clásica de San Sebastián" to answer the following questions.

Rank | Cyclist | Team | Time | UCI ProTour Points
1 | Alejandro Valverde (ESP) | Caisse d'Epargne | 5h 29' 10 | 40
2 | Alexandr Kolobnev (RUS) | Team CSC Saxo Bank | s.t. | 30
3 | Davide Rebellin (ITA) | Gerolsteiner | s.t. | 25
4 | Paolo Bettini (ITA) | Quick Step | s.t. | 20
5 | Franco Pellizotti (ITA) | Liquigas | s.t. | 15
6 | Denis Menchov (RUS) | Rabobank | s.t. | 11
7 | Samuel Sánchez (ESP) | Euskaltel-Euskadi | s.t. | 7
8 | Stéphane Goubert (FRA) | Ag2r-La Mondiale | + 2 | 5
9 | Haimar Zubeldia (ESP) | Euskaltel-Euskadi | + 2 | 3
10 | David Moncoutié (FRA) | Cofidis | + 2 | 1

Question: which country had the most cyclists finish within the top 10?
Explanation: ITA occurs three times in the table, more than any others. Therefore, the answer is Italy.

Question: how many players got less than 10 points?
Explanation: Samuel Sánchez,  Stéphane Goubert, Haimar Zubeldia and David Moncoutié received less than 10 points.  Therefore, the answer is 4.

Question: how many points does the player from rank 3, rank 4 and rank 5 combine to have? 
Explanation: rank 3 has 25 points, rank 4 has 20 points, rank 5 has 15 points, they combine to have a total of 60 points. Therefore, the answer is 60.

Question: who spent the most time in the 2008 Clásica de San Sebastián?
Explanation: David Moncoutié spent the most time to finish the game and ranked the last. Therefore, the answer is David Moncoutié.
"""

if __name__ == "__main__":
	args = parser.parse_args()

	openai.api_key = os.getenv('OPENAI_KEY')

	with open(f'test_qa.json') as f:
		wikitableqa = json.load(f)

	now = datetime.now() 
	dt_string = now.strftime("%d_%H_%M")

	keys = list(wikitableqa.keys())[args.start:args.end]

	correct = 0
	wrong = 0

	if not args.dry_run:
		model_version = args.model.split('-')[1]
		fw = open(f'outputs/response_s{args.start}_e{args.end}_{args.option}_{model_version}_{dt_string}.json', 'w')
		tmp = {'demonstration': demonstration[args.option]}
		fw.write(json.dumps(tmp) + '\n')

	for key in tqdm.tqdm(keys):
		entry = wikitableqa[key]

		question = entry['question']
		answer = entry['answer']

		#### Formalizing the k-shot demonstration. #####
		prompt = demonstration[args.option] + '\n'
		prompt += f'Read the table blow regarding "{entry["title"]}" to answer the following question.\n\n'
		if 'davinci' not in args.model:
			prompt += '\n'.join(entry['table'].split('\n')[:15])
		else:
			prompt += entry['table'] + '\n'
		prompt += 'Question: ' + question + '\nExplanation:'

		if args.dry_run:
			print(prompt)
			print('answer: ', answer)
		else:
			response = openai.Completion.create(
			  model=args.model,
			  prompt=prompt,
			  temperature=0.7,
			  max_tokens=64,
			  top_p=1,
			  frequency_penalty=0,
			  presence_penalty=0
			)

			response = response['choices'][0]["text"].strip().strip('\n').strip('\n').split('\n')[0]

			tmp = {'key': key, 'question': question, 'response': response, 'answer': answer, 'table_id': entry['table_id']}

			fw.write(json.dumps(tmp) + '\n')

	if not args.dry_run:
		print(correct, wrong, correct / (correct + wrong + 0.001))
		fw.close()
