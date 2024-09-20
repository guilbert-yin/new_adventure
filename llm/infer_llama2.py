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
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
import json
import transformers
import argparse
transformers.logging.set_verbosity_info()


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None,
                    help="The directory of the model")
parser.add_argument("--input-fn", type=str, default="mht_dataset_prompt_dev_top30.jsonl", help="Input file name")
parser.add_argument("--output-fn", type=str, default="llama2_output.jsonl", help="Output file name")
parser.add_argument("--infer-num", type=int, default=1044, help="infer dev set num")
parser.add_argument("--use-chat", action="store_true", default=False, help="use model chat interface")
parser.add_argument("--use-raw-prompt", action="store_true", default=False, help="use raw plain prompt template")

args = parser.parse_args()


max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-1}GB'
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}



#model_save_path = '/home/gt/code/git/llama2/tatqa_train_chatmodel_full_output'
# model_ckpt_save_path = '/home/gt/code/git/llama2/tatqa_train_full_output'
model_ckpt_save_path = None

### config ###
#model_id = "/home/gt/huggingface/Llama-2-7b-chat-hf"


# model_id = "/exdata/huggingface/Llama-2-7b-hf"
# model_id = "/exdata/huggingface/Llama-2-7b-chat-hf"
# model_id = "/root/autodl-fs/models/Llama-2-7b-chat-hf"


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



def init_llama_model(model_id):
    # 尝试把这个去掉试一下
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
        max_memory=max_memory,
        # attn_implementation="eager",
        # attn_implementation="flash_attention_2",
        # use_flash_attention_2=True,
        # trust_remote_code=True, # minicpm需要打开这个配置
    )


    # load tokenizer from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    # LoRA Model Configuration
    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM, 
    #     inference_mode=True,
    #     target_modules=['query_key_value'],
    #     r=8, 
    #     lora_alpha=32, 
    #     lora_dropout=0.1
    # )

    peft_config = LoraConfig(
        inference_mode=True,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.eval()

    if model_ckpt_save_path is not None:
        peft_path = f'{model_ckpt_save_path}/checkpoint-100'
        model = PeftModel.from_pretrained(
            model,
            peft_path,
            torch_dtype=torch.float16,
        )
        model.eval()
    
    return tokenizer, model


def infer(fn, tokenizer, model, output_fn, infer_num=1044):
    f = open(fn,"r")
    lines = []
    #with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    for line in f:
        lines.append(line)

    fout = open(output_fn, "w")



    

    with torch.no_grad():
        for l in lines[0:infer_num]:
            js = json.loads(l)
            prompt = js['prompt_prefix']

            if args.use_raw_prompt:
                
                inputs = tokenizer(prompt, max_length=2048, truncation=True, return_tensors="pt")

            else:
                if args.use_chat:
                    model_user_template = f'''
                    <s>[INST] <<SYS>><</SYS>>
                    {prompt} [/INST]
                    '''
                else:
                    model_user_template = f'''
                    <s>{prompt}
                    '''


                inputs = tokenizer(model_user_template, max_length=2048, truncation=True, return_tensors="pt")

            # inputs = tokenizer(prompt, return_tensors="pt")
            generation_output = model.generate(
                    input_ids=inputs.input_ids,
                    generation_config=generation_config,
                    #return_dict_in_generate=True,
                    # max_length=inputs["input_ids"].shape[-1] + 10,
                    max_new_tokens=64,
                    )
            
            res = tokenizer.decode(generation_output[0][inputs["input_ids"].shape[-1]:])
            print("==========")
            print('Answer: ', res)

            dout = {"res":res, "quid": js["quid"]}
            fout.write(json.dumps(dout)+"\n")
            fout.flush()

            torch.cuda.empty_cache()
            print("==========")
            

    f.close()
    fout.close()


# cmd:

# python3 llm/infer_llama2.py --model /exdata/huggingface/Llama-2-7b-chat-hf --input-fn mht_dataset_table_raw_prompt_dev_top30.jsonl --output-fn llama2-chat_table_raw_prompt_refine_output.jsonl --infer-num 50 --use-chat

if __name__ == "__main__":
    model_id = args.model
    output_fn = args.output_fn
    input_fn = args.input_fn
    infer_num = args.infer_num

    tokenizer, model = init_llama_model(model_id)
    infer(input_fn, tokenizer, model, output_fn, infer_num)