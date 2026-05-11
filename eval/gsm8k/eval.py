import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

ANSWER_PROMPT = "The final answer is: "
def extract_answer_number(sentence: str) -> float:
    import re
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default='')
    args = parser.parse_args()
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data_lst = json.load(f)
    output_lst= []
    correct = 0
    total=0
    for data in data_lst:
        answer_ground_truth = extract_answer_number(data['ground_truth'])
        answer = extract_answer_number(data['output'])
        if answer_ground_truth==answer:
            correct +=1 
            data["correct"] ="true"
        else:
            data["correct"] ="false"
        total += 1
        output_lst.append(data)
    print("{:.2f}".format(correct/total*100))
    output_lst .append("score={:.2f}".format(correct/total*100))
    with open(f'{args.input_path}_classification.json', 'w') as f:
        json.dump(output_lst, f, indent=4)
