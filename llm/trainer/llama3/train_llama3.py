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
# from trl import SFTTrainer
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
import transformers
transformers.logging.set_verbosity_info()



### config ###
#model_id = "/home/gt/huggingface/Llama-2-7b-chat-hf"
model_id = "/exdata/huggingface/Llama-2-7b-hf"
max_length = 768
device_map = "cuda:0"
batch_size = 8
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size


tokenizer = None



def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
    
    if param.requires_grad:
        trainable_model_params += param.numel()
    print(f"all params num: {all_model_params}, trainable param num: {trainable_model_params}")
    return trainable_model_params





def init_model(model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,   # load the model into memory using 4-bit precision
        bnb_4bit_use_double_quant=True, # use double quantition
        bnb_4bit_quant_type="nf4", # use NormalFloat quantition
        bnb_4bit_compute_dtype=torch.bfloat16 # use hf for computing when we need
    )

    # load model from huggingface
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=False,
        device_map=device_map
    )


    ori_p = print_number_of_trainable_model_parameters(model)


    model = prepare_model_for_kbit_training(model)
    '''
    - r, the dim of the low_rank matrices
    - lora_alpha, scaling factor, the weight is scaled by lora_alpha/r, 
    the higher value assigns more weight to the LoRA activations
    - target_modules: default is "q_proj", "v_proj"
    - bias, the recommend setting bias to None first, and then lora_only, before trying all.
    '''
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    ### compare trainable parameters
    peft_p = print_number_of_trainable_model_parameters(model)


    # load tokenizer from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer










### generate prompt based on template ###
# prompt_template = {
#    "prompt_input": 
#    "Below is an instruction that describes a task, paired with an input that provides further context.Write a response that appropriately completes the request.\n\n{instruction}\n\n{input}\n### Response:\n",
   
#    "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\\n\\n### Instruction:\\n{instruction}\\n\\n### Response:\\n",
   
#    "response_split": "### Response:"
# }

# def generate_prompt(instruction, input=None, label=None, prompt_template=prompt_template):
#     if input:
#         res = prompt_template["prompt_input"].format(
#             instruction=instruction, input=input)
#     else:
#         res = prompt_template["prompt_no_input"].format(
#             instruction=instruction)
#     if label:
#         res = f"{res}{label}"
#     return res


# def tokenize(tokenizer, prompt, max_length=max_length, add_eos_token=False):
#     result = tokenizer(
#         prompt,
#         truncation=True,
#         max_length=max_length,
#         padding=False,
#         return_tensors=None)

#     result["labels"] = result["input_ids"].copy()
#     return result


def tokenize(tokenizer, prompt, max_length=max_length, add_eos_token=True):
    result = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < max_length
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        
    result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result



def generate_and_tokenize_prompt(data_point):
    full_prompt = data_point['prompt_full']
    tokenized_full_prompt = tokenize(tokenizer, full_prompt)
    return tokenized_full_prompt


# def generate_and_tokenize_prompt(data_point):
#     full_prompt = generate_prompt(
#         data_point["instruction"],
#         data_point["context"],
#         data_point["response"],
#     )
  
#     tokenized_full_prompt = tokenize(tokenizer, full_prompt)
#     # print(f'full prompt: {full_prompt}, \\ntokenized_full_prompt: {tokenized_full_prompt}')
    
#     # user prompt has no response
#     user_prompt = generate_prompt(data_point["instruction"], data_point["context"])
#     tokenized_user_prompt = tokenize(tokenizer, user_prompt)
#     #print(f'\\nuser prompt: {user_prompt}, tokenized_user_prompt: {tokenized_user_prompt}')

#     user_prompt_len = len(tokenized_user_prompt["input_ids"])
#     # -110 means to ignore this token when computing loss
#     mask_token = [-100] * user_prompt_len
#     # print('\\n' + f'mask token: {mask_token}, len: {len(mask_token)}')

#     tokenized_full_prompt["labels"] = mask_token + tokenized_full_prompt["labels"][user_prompt_len:]
#     # print('\\n' + f'tokenized_full_prompt: {tokenized_full_prompt}')
#     return tokenized_full_prompt


def init_dataset(dataset_fn):

    dataset = datasets.load_dataset(
        dataset_fn, split='train'
    )
    dataset = dataset.train_test_split(test_size=3000, shuffle=True, seed=42)
    # cols = ["instruction", "context", "response", "category"]
    cols = ["answer", "response"]

    # train_data = dataset["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols)
    # val_data = dataset["test"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols,)
    train_data = dataset["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols, )
    val_data = dataset["test"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols, )

    return train_data, val_data





if __name__ == "__main__":
    model, tk = init_model(model_id)
    tokenizer = tk

    train_data, val_data = init_dataset("dataset/to_train")
    
    model_save_path = 'mht_train_output'

    args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=20,
        max_steps=400,
        fp16=True,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        group_by_length=False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=10,
        #save_total_limit=50,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=args,
        data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )

    # silence the warnings. re-enable for inference!
    model.config.use_cache = False
    IS_RESUME = False

    if IS_RESUME:
        trainer.train(f'{model_save_path}/checkpoint-200')
    else:
        trainer.train()

    model.save_pretrained("llama-7b-base-int4-mht")
    print('model train is finished')


