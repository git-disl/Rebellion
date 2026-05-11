import os
import argparse

parser = argparse.ArgumentParser(description='Show Accuracy')
parser.add_argument('-i', '--input_dir', help="input dir", required=True)

args = parser.parse_args()
input_dir = args.input_dir


def show_acc():
    res_map = {}
    res_list=[]
    with open(input_dir, "r", encoding="utf8") as reader:
        for line  in reader:
            if "sound :" in line:
                percent = line.strip().split(" ")[2]
                res_map["sound"] = percent
            elif "music :" in line:
                percent = line.strip().split(" ")[2]
                res_map["music"] = percent
            elif "speech :" in line:
                percent = line.strip().split(" ")[2]
                res_map["speech"] = percent
            elif "Total Accuracy:" in line:
                percent = line.strip().split(" ")[2]
                res_map["Total Accuracy"] = percent
    res_list.append(res_map)
    res_list = sorted(res_list, key=lambda x: x["Total Accuracy"], reverse=True)
    header = res_list[0].keys()
    rows = [x.values() for x in res_list]
    print(rows)


show_acc()