import openai
from transformers import pipeline
import llama
import os

def get_evaluator(model_path, eval_prompt):
    pipe = pipeline(model=model_path, device_map="auto")
    def evaluate(**kwargs):
        evaluation = pipe(
            eval_prompt.format(**kwargs),
            max_length=512,
            #do_sample=True,
            #temperature=0.2
        )
        return evaluation[0]["generated_text"]
    return evaluate

def get_gpt_evaluator(model_name, eval_prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    def evaluate(**kwargs):
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                {"role": "user", "content": eval_prompt.format(**kwargs)}
            ]
        )
        return completion["choices"][0]["message"]["content"]
    return evaluate


EVAL_MODEL = "lmsys/vicuna-33b-v1.3"

EVAL_PROMPT = """[User Question]: {prompt}\n\n
[Assistant Response]: {assistant_response}\n
[Correct Response]: {correct_response}\n\n
We would like to request your feedback on the performance of an AI assistant in response to the user question displayed above. 
The user asks the question on observing an image. The assistant's response is followed by the correct response.
\nPlease evaluate the assistant's response based on how closely it matches the correct response which describes tactile feelings. Please compare only the semantics of the answers. DO NOT consider grammatical errors in scoring the assistant. The assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only one value indicating the score for the assistant. \nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.\n\n
"""

def load_model(model_path, llama_dir, args):
    if hasattr(args, "llama_type"):
        llama_type = args.llama_type
        return llama.load(model_path, llama_dir, llama_type=llama_type, args=args)
    return llama.load(model_path, llama_dir, args=args)