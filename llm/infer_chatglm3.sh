python3 llm/infer_chatglm3.py --model /home/gt/huggingface/chatglm3-6b-base --lora-path /home/gt/code/git/ChatGLM3/finetune_basemodel_demo/basemodel-tatqa-bs4-20240103-213604-1e-4/checkpoint-15000/pytorch_model.pt





python3 llm/infer_chatglm3.py --model /home/gt/huggingface/chatglm3-6b --input-fn mht_dataset_prompt_dev_top30.jsonl --output-fn chatglm3_output.jsonl



# 使用refine过的table raw格式的prompt进行infer测试, 只验证前面50条
python3 llm/infer_chatglm3.py --model /home/gt/huggingface/chatglm3-6b --input-fn mht_dataset_table_raw_prompt_dev_top30.jsonl --output-fn chatglm3_table_raw_prompt_refine_output.jsonl --infer-num 50

# chatglm3-base
python3 llm/infer_chatglm3.py --model /home/gt/huggingface/chatglm3-6b-base --input-fn mht_dataset_table_raw_prompt_dev_top30.jsonl --output-fn chatglm3-base_table_raw_prompt_refine_output.jsonl --infer-num 50 --use-chat


# use model chat interface
python3 llm/infer_chatglm3.py --model /home/gt/huggingface/chatglm3-6b-base --input-fn mht_dataset_table_raw_prompt_dev_top30.jsonl --output-fn chatglm3-base_chat_table_raw_prompt_refine_output.jsonl --infer-num 50 --use-chat




python3 llm/infer_chatglm3.py --model /home/gt/huggingface/chatglm3-6b-base --input-fn mht_dataset_table_raw_prompt_example1_dev_top30.jsonl --output-fn chatglm3-base_table_raw_prompt_example1_refine_output.jsonl --infer-num 50


python3 llm/infer_chatglm3.py --model /home/gt/huggingface/chatglm3-6b --input-fn mht_dataset_table_raw_prompt_example1_dev_top30.jsonl --output-fn chatglm3-base_table_raw_prompt_example1_refine_output.jsonl --infer-num 50



python3 llm/infer_chatglm3.py --model /home/gt/huggingface/chatglm3-6b-base --input-fn mht_dataset_table_raw_prompt_example1_dev_top30.jsonl --output-fn chatglm3-base_table_raw_prompt_example1_refine_output.jsonl --infer-num 50



python3 llm/infer_chatglm3.py --model /home/gt/huggingface/chatglm3-6b-base --input-fn mht_dataset_table_raw_prompt_example3_dev_top30.jsonl --output-fn chatglm3-base_table_raw_prompt_example3_refine_output_full.jsonl --infer-num 1044 --max-new-tokens 512