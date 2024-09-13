import os, sys
import torch
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    GenerationConfig
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
import json

import transformers
transformers.logging.set_verbosity_info()

#model_save_path = '/home/gt/code/git/llama2/tatqa_train_chatmodel_full_output'
# model_ckpt_save_path = '/home/gt/code/git/llama2/tatqa_train_full_output'
model_ckpt_save_path = None

### config ###
#model_id = "/home/gt/huggingface/Llama-2-7b-chat-hf"


# model_id = "/exdata/huggingface/Llama-2-7b-hf"
model_id = "/exdata/huggingface/Llama-2-7b-chat-hf"
max_length = 512
device_map = "cuda:0"
batch_size = 8
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size


generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4, # beam search
)



def tokenize(tokenizer, prompt, max_length=max_length, add_eos_token=False):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None)

    result["labels"] = result["input_ids"].copy()
    return result



def init_llama_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,   # load the model into memory using 4-bit precision
        bnb_4bit_use_double_quant=True, # use double quantition
        bnb_4bit_quant_type="nf4", # use NormalFloat quantition
        bnb_4bit_compute_dtype=torch.bfloat16 # use hf for computing when we need
    )


    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=True,
        device_map='auto',
        # attn_implementation="eager",
        use_flash_attention_2=True,
    )


    # load tokenizer from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    if model_ckpt_save_path is not None:
        peft_path = f'{model_ckpt_save_path}/checkpoint-100'
        model = PeftModel.from_pretrained(
            model,
            peft_path,
            torch_dtype=torch.float16,
        )
        model.eval()
    
    return tokenizer, model


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
            generation_output = model.generate(
                    input_ids=inputs.input_ids,
                    generation_config=generation_config,
                    #return_dict_in_generate=True,
                    max_length=inputs["input_ids"].shape[-1] + 10,
                    #max_new_tokens=64,
                    )
            print('Answer: ', tokenizer.decode(generation_output[0][inputs["input_ids"].shape[-1]:]))

    f.close()

if __name__ == "__main__":
    tokenizer, model = init_llama_model()
    infer("mht_dataset_prompt_dev_top30.jsonl", tokenizer, model)