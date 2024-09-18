import argparse
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import os
from peft import get_peft_model, LoraConfig, TaskType
import json


# Argument Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None,
                    help="The directory of the model")
parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path")
parser.add_argument("--lora-path", type=str, default=None,
                    help="Path to the LoRA model checkpoint")
parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation")
parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum new tokens for generation")
parser.add_argument("--lora-alpha", type=float, default=32, help="LoRA alpha")
parser.add_argument("--lora-rank", type=int, default=8, help="LoRA r")
parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
parser.add_argument("--input-fn", type=str, default="mht_dataset_prompt_dev_top30.jsonl", help="Input file name")
parser.add_argument("--output-fn", type=str, default="glm3_output.jsonl", help="Output file name")
parser.add_argument("--infer-num", type=int, default=1044, help="infer dev set num")

args = parser.parse_args()

if args.tokenizer is None:
    args.tokenizer = args.model

# Model and Tokenizer Configuration
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, encode_special_tokens=True)
model = AutoModel.from_pretrained(args.model, load_in_8bit=False, trust_remote_code=True, device_map="auto").to(
    args.device)

# LoRA Model Configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=True,
    target_modules=['query_key_value'],
    r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
)
model = get_peft_model(model, peft_config)
if args.lora_path != None and os.path.exists(args.lora_path):
    model.load_state_dict(torch.load(args.lora_path), strict=False)

model.eval()



# Interactive Prompt

# fin = open("xxxx.jsonl","r")
# fout = open("out.jsonl","w")



# # Interactive Prompt
# for line in fin:
#     jsin = json.loads(line)
#     prompt = jsin['prompt']
#     inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
#     response = model.generate(input_ids=inputs["input_ids"],max_length=inputs["input_ids"].shape[-1] + args.max_new_tokens)
#     response = response[0, inputs["input_ids"].shape[-1]:]
#     res = tokenizer.decode(response, skip_special_tokens=True)
#     dout = {"res":res, "q_uid": jsin["question_uid"]}
#     fout.write(json.dumps(dout)+"\n")
#     print("Response:", res)

# fin.close()
# fout.close()



def infer(fn, tokenizer, model, output_fn, infer_num):
    f = open(fn,"r")
    lines = []
    #with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    for line in f:
        lines.append(line)

    fout = open(output_fn,"w")

    with torch.no_grad():
        for l in lines[0:infer_num]:
            js = json.loads(l)
            prompt = js['prompt_prefix']

            inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

            inputs = inputs.to(args.device)

            gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
            response = model.generate(**inputs, **gen_kwargs)

            response = response[:, inputs["input_ids"].shape[1]:]
            res = tokenizer.decode(response[0], skip_special_tokens=True)


            dout = {"res":res, "quid": js["quid"]}
            fout.write(json.dumps(dout)+"\n")

            print("==========")
            print("Response: ", res)
            print("==========")

            fout.flush()

            
            torch.cuda.empty_cache()
            
            

    f.close()
    fout.close()


input_file_name = args.input_fn
output_file_name = args.output_fn
infer_num = args.infer_num

infer(input_file_name, tokenizer, model, output_file_name, infer_num)

# cmd:
# python3 llm/infer_chatglm4.py --model /exdata/huggingface/glm-4-9b-chat --input-fn mht_dataset_table_raw_prompt_dev_top30.jsonl --output-fn chatglm4_table_raw_prompt_refine_output.jsonl --infer-num 50