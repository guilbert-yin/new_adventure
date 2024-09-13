import json



# def fillin_prompt_template(pt):



def get_prompt_template(fn):
    pf = open(fn, 'r')
    return pf


def get_table(table):
    table_content = ""

    header = table[0]
    new_header = [r if r != "" else "---" for r in header]
    header_str = "|".join(new_header)
    header_str = "|" + header_str + "|\n"

    table_content += header_str

    for j in range(1,len(table)):
        row = table[j]
        new_row = [r if r != "" else "---" for r in row]
        row_str = "|".join(new_row)
        row_str = "|" + row_str + "|\n"

        table_content += row_str
    
    return table_content


def get_para(para):
    pc = ""
    for p in para:
        c = p['text']
        pc += c + "\n"
    return pc





# Please organize the results in the following table:
# | step | output |
# | 1 | {question_type} |
# | 2 | {evidence} |
# | 3 | {equation} |
# | 4 | {answer} |
# | 5 | {scale} |
# Finally, present the final answer in the format: "The answer is: {answer} #### and its corresponding scale is: {scale}"

# ### Table
# {table}

# ### Text
# {text}

# ### Question
# {question}

# ### Response
# | step | output |
# | 1 | {gold_question_type} |
# | 2 | {gold_evidence} |
# | 3 | {gold_equation} |
# | 4 | {gold_answer} |
# | 5 | {gold_scale} |
# The answer is: {gold_answer} #### and its corresponding scale is: {gold_scale}


if __name__ == "__main__":

    # fout = open('llama_dataset_train.jsonl','w')
    pf_header = get_prompt_template("llm/prompts/tat_prompt_header.txt")
    pf_body = get_prompt_template("llm/prompts/tat_prompt_body.txt")
    prompt_template = pf_body.read()


    f = open('llm/dataset/tat-qa/tatqa_dataset_dev.json','r')
    l = json.loads(f.read())

    i = 0

    final_list = []
    max_length = 0
    for d in l:
        i += 1
        print("--------------")
        table = d['table']['table']
        paras = d['paragraphs']
        questions = d['questions']

        table_str = get_table(table)
        paras_str = get_para(paras)

        for q in questions:
            t = q['answer_type']
            source=q['answer_from']
            
            if t == "arithmetic":
                derivation=q['derivation']
            else:
                derivation = 'N.A.'
            
            question=q['question']
            a = q['answer']
            a_str = ''

            if isinstance(a, list):
                a_str = ",".join(a)
            elif isinstance(a, str):
                a_str = a
            else:
                a_str = str(a)

            prompt_body_str = prompt_template.replace("{question_type}", t)
            print(prompt_body_str)

        

        


        # print(context_str)
        # out_d = {"category":"closed_qa", "context": context_str, "instruction": instru, "response": ans}
        # out_str = json.dumps(out_d)
        # fout.write(out_str + "\n")


    print(i)
    print(max_length)
    print(sum(final_list)/len(final_list))