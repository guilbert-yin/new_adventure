
python3 llm/infer_llama2.py --model /exdata/huggingface/Llama-2-7b-chat-hf --input-fn mht_dataset_prompt_dev_top30.jsonl --output-fn llama2-chat_output.jsonl --infer-num 50


# 使用refine过的table raw格式的prompt进行infer测试, 只验证前面50条
python3 llm/infer_llama2.py --model /exdata/huggingface/Llama-2-7b-chat-hf --input-fn mht_dataset_table_raw_prompt_dev_top30.jsonl --output-fn llama2-chat_table_raw_prompt_refine_output.jsonl --infer-num 50 --use-chat


python3 llm/infer_llama2.py --model /exdata/huggingface/Llama-2-7b-chat-hf --input-fn mht_dataset_table_raw_prompt_dev_top30.jsonl --output-fn llama2-chat_table_raw_prompt_refine_output.jsonl --infer-num 50 --use-raw-prompt


# llama2
python3 llm/infer_llama2.py --model /exdata/huggingface/Llama-2-7b-hf --input-fn mht_dataset_table_raw_prompt_dev_top30.jsonl --output-fn llama2_table_raw_prompt_refine_output.jsonl --infer-num 50 --use-raw-prompt