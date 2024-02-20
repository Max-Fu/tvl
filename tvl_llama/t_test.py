import argparse 
import numpy as np 
import json 

def get_scores(json_fp, dataset : str = None):
    with open(json_fp, "r") as f:
        data = json.load(f)
    
    if dataset is not None:
        if dataset == "ssvtp":
            keyword = "images_rgb"
        elif dataset == "hct":
            keyword = "/vision/"
        else:
            raise NotImplementedError("Dataset not supported")
        scores = [float(i["evaluation"].split()[0]) for i in data if keyword in i["image_fp"]]
    else:
        scores = [float(i["evaluation"].split()[0]) for i in data]
    return np.array(scores)

def get_scores_text(text_fp):
    scores = []
    with open(text_fp, "r") as f:
        save_next = False
        for line in f:
            if save_next:
                scores.append(float(line.strip()))
                save_next = False
            if line.startswith("GROUND TRUTH"):
                save_next = True
    return np.array(scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tvl_path", type=str, required=True)
    parser.add_argument("--alt_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    if args.tvl_path.endswith(".json"):
        vtl_scores = get_scores(args.tvl_path, args.dataset)
    else:
        vtl_scores = get_scores_text(args.tvl_path)
    if args.alt_path is not None:
        gpt4v_scores = get_scores(args.alt_path, args.dataset)
    else:
        gpt4v_scores = get_scores("results/gpt4v.json", args.dataset)

    print("VTL mean: ", np.mean(vtl_scores))
    print("GPT4V mean: ", np.mean(gpt4v_scores))

    # running t test 
    from scipy.stats import ttest_1samp
    sample = vtl_scores - gpt4v_scores
    t, p = ttest_1samp(sample, 0)
    print("t: ", t)
    print("p: ", p)