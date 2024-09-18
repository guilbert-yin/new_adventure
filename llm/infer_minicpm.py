from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="/exdata/huggingface/models--openbmb--MiniCPM3-4B",
                    help="The directory of the model")
parser.add_argument("--input-fn", type=str, default="mht_dataset_prompt_dev_top30.jsonl", help="Input file name")
parser.add_argument("--output-fn", type=str, default="minicpm_output.jsonl", help="Output file name")
parser.add_argument("--infer-num", type=int, default=1044, help="infer dev set num")

args = parser.parse_args()


def init_llama_model(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    return tokenizer, model


def infer(fn, tokenizer, model, output_fn, infer_num):
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
            quid = js['quid']



            messages = [
                {"role": "user", "content": prompt},
            ]
            
            model_inputs = tokenizer.apply_chat_template(messages, truncation=True, max_length=4096, return_tensors="pt", add_generation_prompt=True).to(device)


            attention_mask = torch.ones(model_inputs.shape,dtype=torch.long,device=device)

            model_outputs = model.generate(
                model_inputs,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=128,
                top_p=0.7,
                temperature=0.2
            )

            output_token_ids = [
                model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
            ]

            responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
            print("LLM response: ")
            print(responses)
            print("=====================")

            dout = {"quid":quid, "res":responses}
            fout.write(json.dumps(dout)+"\n")
            fout.flush()

            torch.cuda.empty_cache()



    fout.close()
    f.close()





# install pkgs:
# pip install datamodel_code_generator jsonschema
# pip install modelscope accelerate transformers


# cmd
# python3 llm/infer_minicpm.py --model /exdata/huggingface/models--openbmb--MiniCPM3-4B --input-fn mht_dataset_table_raw_prompt_dev_top30.jsonl --output-fn minicpm_table_raw_prompt_refine_output.jsonl --infer-num 50



if __name__ == "__main__":
    # model_id = "/exdata/huggingface/models--openbmb--MiniCPM3-4B"
    device = "cuda"

    model_id = args.model
    output_fn = args.output_fn
    input_fn = args.input_fn
    infer_num = args.infer_num

    tokenizer, model = init_llama_model(model_id, device)
    infer(input_fn, tokenizer, model, output_fn, infer_num)