import json
import sys
import glob
import argparse
from collections import Counter
import pprint
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))
import show_context

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", required=True, type=str)
parser.add_argument("--cutoff", default=0, type=int)
parser.add_argument("--error", default=False, action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()

    with open('../test_statements_all.json', 'r') as f:
        tables = json.load(f)

    counter = {}
    for filename in glob.glob(args.inputs):
        label_counter = Counter()
        label_counter_ref = Counter()
        correct = 0
        wrong = 0
        with open(filename, 'r') as f:
            for line in f:
                response = json.loads(line.strip())
                # independent predictions
                if 'prediction' not in response:
                    continue

                table_length = len(tables[response['key']]['table'].split(' ')) // 100
                if table_length not in counter:
                    counter[table_length] = {'correct': 0, 'wrong': 0}

                if response['label'] == response['prediction']:
                    correct += 1
                    counter[table_length]['correct'] += 1
                else:
                    wrong += 1
                    counter[table_length]['wrong'] += 1
                    if args.error:
                        show_context.show_context(response['key'])
                        pprint.pprint(response)
                        print()
                label_counter.update([response['prediction']])
                label_counter_ref.update([response['label']])

                if correct + wrong >= args.cutoff and args.cutoff > 0:
                    break

        print(filename)
        print('prediction', label_counter, 'reference', label_counter_ref)
        print('accuracy: ', correct / (correct + wrong), 'total example: ', correct + wrong)

    counter = sorted(counter.items(), key=lambda x: x[0])
    for k, v in counter:
        print(k, v['correct'] / (v['correct'] + v['wrong']))