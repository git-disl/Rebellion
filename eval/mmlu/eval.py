import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re

def extract_answer_number(sentence: str) -> float:
    # Remove commas from the sentence as per the original partial code
    sentence = sentence.replace(',', '')

    # Define a mapping from option letters to their numerical values
    # Initialize the prediction answer. Default to 0.0 if no options are found.
    pred_answer = 0.0

    # Find all occurrences of 'A', 'B', 'C', or 'D' in the sentence.
    # re.findall returns a list of all non-overlapping matches.
    # re.IGNORECASE makes the search case-insensitive, matching 'a' as well as 'A'.
    found_options = re.findall(r'[ABCD]', sentence)

    # If any options were found, we need to select the final one
    if found_options:
        # The last element in the `found_options` list is the final one encountered.
        # Convert it to uppercase to ensure it matches the keys in `option_map`.
        final_option_char = found_options[-1].upper()
        # Get the numerical value from the map.
        # Use .get() with a default of 0.0 in case an unexpected character somehow
        # ends up here (though the regex should prevent this).
    else:
        final_option_char = None

    return final_option_char


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
        answer_ground_truth = data['ground_truth']
        answer = extract_answer_number(data['output'])
        # print(answer)
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
