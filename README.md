# TableCoT
The code and data used for EACL-2023 Paper [Large Language Models are few(1)-shot Table Reasoners](https://arxiv.org/abs/2210.06710)


## Preliminary
First, you need to specify your OPENAI_API_KEY, please find it in your account in https://openai.com/api/.
```
export OPENAI_KEY=[YOUR_KEY]
```

## For WikiTableQuestions
```
python prompt.py --start 0 --end 500
```
This will call Chain of Thoughts prompting to solve the 0-500 example in the test set of WikiTableQA. The output will be saved to output/response_..._s0_e500.json.

You can further call this following to extract the answers from the predictions.
```
cd outputs/
python postprocess_answer.py --inputs response_..._s0_e500.json
```

Finally, call this following to compute the final EM score.
```
python compute_scores.py --inputs response_..._s0_e500.json.processed
```

## For TabFact
```
python prompt.py --start 0 --end 500
```
This will call Chain of Thoughts prompting to solve the 0-500 example in the test set of WikiTableQA. The output will be saved to output/response_..._s0_e500.json. This will directly output the accuracy after it finishes.

