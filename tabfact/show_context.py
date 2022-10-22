import sys
import json
import pprint
from os.path import dirname


def show_context(index):
	folder = dirname(__file__)
	with open(f'{folder}/test_statements_simple.json') as f:
		dataset = json.load(f)
	with open(f'{folder}/test_statements_complex.json') as f:
		dataset.update(json.load(f))

	print('Title: ', dataset[index]['title'])

	print(dataset[index]['table'])

	if 'context' in dataset[index]: 
		pprint.pprint(dataset[index]['context'])

if __name__ == '__main__':
	show_context(sys.argv[1])
