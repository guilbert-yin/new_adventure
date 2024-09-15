from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import json





def init_llama_model(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
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



            messages = [
                {"role": "user", "content": prompt},
            ]
            model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

            model_outputs = model.generate(
                model_inputs,
                max_new_tokens=1024,
                top_p=0.7,
                temperature=0.7
            )

            output_token_ids = [
                model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
            ]

            responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
            print("LLM response: ")
            print(responses)
            print("=====================")




    f.close()





# install pkgs:
# pip install datamodel_code_generator jsonschema
# pip install modelscope accelerate transformers



if __name__ == "__main__":
    model_id = "/exdata/huggingface/models--openbmb--MiniCPM3-4B"
    device = "cuda"

    tokenizer, model = init_llama_model(model_id, device)
    infer("mht_dataset_prompt_dev_top30.jsonl", tokenizer, model)