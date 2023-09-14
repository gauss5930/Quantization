import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default="/inference/inference_result.json")

    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.file_path, 'r') as f:
        data = json.load(f)

    sns.set(style="darkgrid")

    sns.lineplot(data=data[0], color="blue", label="bnb-bnb")
    sns.lineplot(data=data[1], color="orange", label="gptq-gptq")
    sns.lineplot(data=data[2], color="green", label="bnb-gptq")

    plt.ylabel("Average inference time (s)")
    plt.xlabel("Batch size")
    
    plt.legend()

if __name__ == "__main__":
    main()