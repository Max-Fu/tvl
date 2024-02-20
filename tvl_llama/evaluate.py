# example: OPENAI_API_KEY=<openai-key> CUDA_VISIBLE_DEVICES=0 python evaluate_all.py --has_lora --model_path ~/Documents/meta/tvl-hf/ckpt/tvl_llama/tvl_llama_vits.pth --gpt --active_modality_names tactile vision --tactile_model vit_small_patch16_224 --eval_datasets ssvtp hct --datasets_dir ~/dataset/tvl_dataset
import os
import argparse
from tvl_enc import tacvis
import json
import csv
from util.eval_util import get_evaluator, get_gpt_evaluator, EVAL_MODEL, EVAL_PROMPT, load_model
import llama

def setup_parser():
    parser = argparse.ArgumentParser('eval', add_help=False)
    parser.add_argument("--datasets_dir", type=str, help="Directory containing the datasets")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the LLaMA-adapter Checkpoint")
    parser.add_argument('--llama_type', type=str, default="llama-2-7b")
    parser.add_argument('--llama_dir', type=str, default="/home/mfu/checkpoints/llama-2")
    parser.add_argument('--has_lora', action='store_true', help='Whether to use LoRA layers.')
    parser.add_argument('--crop_tacvis', action='store_true', help='whether to crop image observations')
    parser.add_argument("--active_modality_names", type=str, nargs="+", default=["tactile", "vision"],
                        help="List of active modalities.")
    parser.add_argument("--lora_modality_names", type=str, nargs="+", default=[],
                        help="List of active modalities.")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of LoRA layers")
    parser.add_argument("--lora_layer_idxs", nargs="+", type=int, help="Layer indices to apply LoRA")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the tvl-encoder checkpoint.")
    parser.add_argument("--gpt", action="store_true", help="Use GPT-4 to eval (requires OPENAI_API_KEY)")
    parser.add_argument("--background_sub", action="store_true", help="Use background sub for tacvis")
    parser.add_argument("--eval_datasets", type=str, nargs="+", default=["ssvtp"], help="List of active modalities.", choices=["ssvtp", "hct"])
    parser.add_argument('--tactile_model', type=str, default='resnet18', choices=["vit_base_patch16_224", "vit_small_patch16_224", "vit_tiny_patch16_224", "resnet18"],  help="Tactile encoder model")
    parser.add_argument("--resume_from_json", type=str, default=None, help="Path to resume evaluation")
    return parser.parse_args()

def get_valid_img_file_path():
    image_fp = input("Provide file path: ")
    while not os.path.exists(image_fp):
        image_fp = input("Incorrect file path, provide file path again: ")
    return image_fp

def get_valid_tactile_file_path():
    adding_tactile = input("Add tactile? (y/n): ").lower() == "y"
    if not adding_tactile:
        return None
    image_fp = input("Provide file path: ")
    while not os.path.exists(image_fp):
        image_fp = input("Incorrect file path, provide file path again: ")
    return image_fp

def main():
    # Tactile related LoRA
    args = setup_parser() 
    model_path = args.model_path 
    llama_dir = args.llama_dir

    use_gpt = args.gpt
    background_sub = args.background_sub
    del args.background_sub
    del args.gpt
    
    # Load the model
    model = load_model(model_path, llama_dir, args)
    model.eval()

    if use_gpt:
        eval_fn = get_gpt_evaluator("gpt-4", EVAL_PROMPT)
    else:
        eval_fn = get_evaluator(EVAL_MODEL, EVAL_PROMPT)
    
    ratings = []
    images = []
    labels = []
    tactiles = []
    backgrounds = []
    dataset_version = []
    evaluated_images = []
    eval_data = []
    out_json = args.resume_from_json
    output_json_path = os.path.basename(os.path.dirname(args.model_path)) + "_{}_".format("_".join(args.active_modality_names)) + "eval_data.json"

    if out_json is not None:
        assert os.path.exists(out_json), f"{out_json} does not exist"
        with open(out_json, "r") as f:
            eval_data = json.load(f)
            for d in eval_data:
                evaluated_images.append(d["image_fp"])
                ratings.append(d["evaluation"])

    # all tacvis test set 
    if "ssvtp" in args.eval_datasets:
        ssvtp_dir = os.path.join(args.datasets_dir, "ssvtp")
        with open(os.path.join(ssvtp_dir, "test.csv"), newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                img = row[0]
                img = os.path.join(ssvtp_dir, img)
                if img in evaluated_images:
                    continue
                images.append(img)
                tactiles.append(img.replace("rgb", "tac"))
                backgrounds.append(tacvis.TAC_BG_FP)
                labels.append(row[1])
                dataset_version.append("v1")

    if "hct" in args.eval_datasets:
        hct_dir = os.path.join(args.datasets_dir, "hct")
        m3_test_csvs = [
            os.path.join(hct_dir, "data2/test.csv"), 
            os.path.join(hct_dir, "data3/test.csv"), 
            os.path.join(hct_dir, "data1/test.csv"), 
        ]
        for csv_path in m3_test_csvs:
            folder_path = os.path.dirname(csv_path)
            with open(csv_path, newline='') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    img = row[0]
                    img = os.path.join(folder_path, img)
                    if img in evaluated_images:
                        continue
                    images.append(img)
                    tactiles.append(os.path.join(folder_path, row[1]))
                    backgrounds.append(os.path.join(folder_path, row[2]))
                    labels.append(row[3])
                    dataset_version.append("v2")

    prompt = "This image gives tactile feelings of?"
    for idx, (image_fp, tactile_fp, background, label, dv) in enumerate(zip(images, tactiles, backgrounds, labels, dataset_version)):
        inputs = {}
        if "vision" in args.active_modality_names:
            image = tacvis.load_vision_data(image_fp, device='cuda', dataset_version=dv).unsqueeze(0)
            inputs['vision'] = [image, 1]
        
        if "tactile" in args.active_modality_names:
            if background_sub:
                transform_tac = tacvis.tac_subtract_bg(background, tacvis.TAC_MEAN_BG, tacvis.TAC_STD_BG)
                tactile = tacvis.load_tactile_data(tactile_fp, device='cuda', transform_tac=transform_tac)
            else:
                tactile = tacvis.load_tactile_data(tactile_fp, device='cuda')
            tactile = tactile.unsqueeze(0)
            inputs['tactile'] = [tactile, 1]

        print(f"{idx}/{len(images)} Generating for {image_fp}...")
        # Generate results
        results = model.generate(
            inputs,
            [llama.format_prompt(prompt)],
            max_gen_len=256
        )
        assistant_response = results[0].strip()

        print("GROUND TRUTH:", label, "ASSISTANT:", assistant_response)
        evaluation = eval_fn(prompt=prompt, assistant_response=assistant_response, correct_response=label)
        print(evaluation)
        ratings.append(evaluation)
        print("\n\n")
        eval_data.append({
            "image_fp": image_fp,
            "tactile_fp":tactile_fp, 
            "label": label, 
            "generated response": assistant_response,
            "prompt" : "This image gives tactile feelings of?",
            "evaluation": evaluation,
        })
    
        with open(output_json_path, "w") as f:
            json.dump(eval_data, f)
    
    n_evals = 0
    sum_ratings = 0
    for evaluation in ratings:
        try:
            sum_ratings += float(evaluation.split()[0])
        except:
            continue
        n_evals += 1
    print("\n\nAverage rating", sum_ratings / n_evals)


if __name__ == "__main__":
    main()
