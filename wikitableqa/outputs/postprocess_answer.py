import json
import sys
import glob
import argparse
import re
import numpy as np
import os
import openai
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", required=True, type=str)

demonstration = """
answer the following questions based on the evidence.

question: how many people were murdered in 1940/41?
evidence: According to the table, 100,000 people were murdered in 1940/41.
answer: 100,000

question: alfie's birthday party aired on january 19. what was the airdate of the next episode?
evidence: According to the table, the next episode aired on January 26, 1995.
answer: January 26, 1995

question: how many awards has leona lewis won?
evidence: According to the table, Leona Lewis has won nineteen awards in total.
answer: 19

question: what time period had no shirt sponsor?
evidence: 1977-1978 had no shirt sponsor according to the table.
answer: 1977-1978

question: in which competition did hopley finish fist?
evidence: Hopley finished first in the World Junior Championships in 2000.
answer: World Junior Championships
"""


def get_answer(pred):
	match = re.search(r'(The|the) answer is ([^\.]+)\.$', pred)
	if match:
		return match.group(2).strip('"'), True
	return pred, False


if __name__ == "__main__":
	args = parser.parse_args()
	openai.api_key = os.getenv('OPENAI_KEY')

	for filename in glob.glob(args.inputs):
		print('Start', filename)

		fw = open(filename + '.processed', 'w')
		with open(filename, 'r') as f:
			for line in tqdm.tqdm(f):
				response = json.loads(line.strip())
				if 'response' in response:
					prediction = response['response'].strip('\n').strip('\n\n').split('\n')[0]
					prediction, success = get_answer(prediction)

					if success:
						response.pop('response')
						response['prediction'] = prediction
						response['direct'] = True
						fw.write(json.dumps(response) + '\n')
					else:
						prompt = demonstration + '\n'
						prompt += f'question: {response["question"]}\n'
						prompt += f'evidence: {response["response"]}\n'
						prompt += 'answer:'

						tmp = openai.Completion.create(
						  model="text-davinci-002",
						  prompt=prompt,
						  temperature=0.7,
						  max_tokens=64,
						  top_p=1,
						  frequency_penalty=0,
						  presence_penalty=0
						)
						tmp = tmp['choices'][0]["text"].strip().strip('\n')

						response.pop('response')
						response['prediction'] = tmp
						response['extracted'] = True
						fw.write(json.dumps(response) + '\n')

		fw.close()