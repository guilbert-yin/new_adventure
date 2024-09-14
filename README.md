1. 数据生成脚本：llm/data_process_mht.py \
python llm/data_process_mht.py dev # 生成dev数据集的prompt文件 \
输入文件 llm/dataset/mht/combine_reason_input_dev_top30.json \
输出文件 mht_dataset_prompt_dev_top30.jsonl \
mht_dataset_prompt_dev_top30.jsonl 文件的json格式如下：\
```code
{"quid":quid, 
"prompt_full": final_prompt, 
"prompt_prefix":prompt_prefix, 
"program": program, 
"answer": str(answer), 
"response": response_str.replace("####", "")}
```
prompt_full 是用来训练的全部prompt内容 \
prompt_prefix 是用来推理的不包含 #### Reponse 内容的信息 \
response 是只包含 #### Reponse 的内容

2. 推理代码 llm/infer.py \
python llm/infer.py

3. 训练代码 llm/train_mht.py \
python llm/train_mht.py

4. 输入到模型推理的数据文件样例：inference_example.txt \
训练的数据文件样例：train_example.txt

5. flash attention 安装了 flash_attn-2.1.1+cu121torch2.1 \
代码：
```code
  model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=True,
        device_map='auto',
        # attn_implementation="eager",
        use_flash_attention_2=True,
    )
```
但是报：
RuntimeError: FlashAttention only supports Ampere GPUs or newer.