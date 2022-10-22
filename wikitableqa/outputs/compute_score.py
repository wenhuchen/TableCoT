import json
import sys
import glob
import argparse
import re
import numpy as np
import string


parser = argparse.ArgumentParser()
parser.add_argument("--cutoff", default=-1, type=int)
parser.add_argument("--inputs", required=True, type=str)


def maybe_normalize_float(span: str):
    if span and (re.match(r"^[+-][0-9]+[.]?[0-9]*$", span)
                 or (re.match(r"^[0-9]*[.]?[0-9]*$", span))) and span != '.':
        # FIXME: We did this(instead of try except) to convert a string into a float
        #  since the try catch will lead to an error when using 8 V100 gpus with cuda 11.0,
        #  and we still don't know why that could happen....
        return str(float(span))
    else:
        return span


def maybe_normalize_number(text: str) -> str:
    units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    for index, unit in enumerate(units):
        if text == unit:
            return str(float(index))
    return text


def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)


def remove_articles(text: str) -> str:
    return re.sub(r'\b(a|an|the)\b', ' ', text)


def eval_ex_match(pred, gold_result):
    pred = pred.lower()
    gold_result = gold_result.lower()

    # Replace and with comma
    if ' and ' in pred and '|' in gold_result:
        pred = pred.replace(' and ', ', ')

    pred = [span.strip() for span in pred.split(', ')]

    if '|' in gold_result:
        gold_result = [span.strip() for span in gold_result.split('|')]
    else:
        gold_result = [span.strip() for span in gold_result.split(', ')]

    pred = [maybe_normalize_number(remove_punc(remove_articles(span.strip()))) for span in pred]
    gold_result = [maybe_normalize_number(remove_punc(remove_articles(span.strip()))) for span in gold_result]

    # print(pred, ' # ', gold_result)
    clean_float = True  # TODO
    if clean_float:
        pred = [maybe_normalize_float(span) for span in pred]
        gold_result = [maybe_normalize_float(span) for span in gold_result]

    return sorted(pred) == sorted(gold_result)


if __name__ == "__main__":
    args = parser.parse_args()

    with open('../test_qa.json', 'r') as f:
        tables = json.load(f)

    counter = {}
    for filename in glob.glob(args.inputs):
        correct = 0
        wrong = 0
        print('Start', filename)
        with open(filename, 'r') as f:
            for line in f:
                response = json.loads(line)
                if 'prediction' in response:
                    table_length = len(tables[response['key']]['table'].split(' ')) // 100
                    if table_length not in counter:
                        counter[table_length] = {'correct': 0, 'wrong': 0}

                    if eval_ex_match(response['prediction'], response['answer']):
                        correct += 1
                        counter[table_length]['correct'] += 1
                    else:
                        wrong += 1
                        counter[table_length]['wrong'] += 1
                else:
                    continue

                if correct + wrong >= args.cutoff and args.cutoff > -1:
                    break

        print('Done with', filename)
        print('accuracy: ', correct / (correct + wrong), 'correct example: ', correct, 'total example: ', correct + wrong)

    counter = sorted(counter.items(), key=lambda x: x[0])
    for k, v in counter:
        print(k, v['correct'] / (v['correct'] + v['wrong']))