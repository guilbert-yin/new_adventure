import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

model_name = '/root/autodl-fs/models/Llama-2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-1}GB'
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    load_in_8bit=True,
    max_memory=max_memory,
    attn_implementation="flash_attention_2",
    trust_remote_code=True
)
model = model.eval()


def infer(fn, tokenizer, model):
    f = open(fn,"r")
    lines = []
    #with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    for line in f:
        lines.append(line)

    with torch.no_grad():
        for l in lines:
            js = json.loads(l)
            prompt = js['prompt_prefix']
            inputs = tokenizer(prompt, return_tensors="pt")

            generation_output = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096)
            print('--- Answer: ', tokenizer.decode(generation_output[0][inputs["input_ids"].shape[-1]:]))
            print("=======")
            torch.cuda.empty_cache()

    f.close()

if __name__ == "__main__":
    infer("mht_dataset_prompt_dev_top30.jsonl", tokenizer, model)


