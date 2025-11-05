from math_verify import parse, verify
import json
from tqdm import tqdm
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def evaluate(output_text, answer):
    a1 = parse("\\boxed{" + answer + "}", parsing_timeout=5)
    a2 = parse(output_text, parsing_timeout=5)
    if verify(a1, a2):
        return True
    else:
        return False


if __name__ == "__main__":
    args = parse_arguments()
    with open(args.output_path, "r") as f:
        res = json.load(f)
        res = res["data"]
    count = 1e-6
    for i in tqdm(range(len(res))):
        if evaluate(res[i]["output"], res[i]["answer"]):
            count += 1
    print("pass@1: ", count / len(res))
