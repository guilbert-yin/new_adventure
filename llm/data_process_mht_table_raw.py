import json
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import sys
import re

def to_df(html_table):
    soup = BeautifulSoup(html_table, 'html.parser')
    table = soup.find('table')
    # 将表格转换为数据框，并将跨列单元格替换为空值
    df = pd.read_html(str(table))[0]
    return df

def table_to_2d_array(html_table):
    """
    将HTML表格转换为二维数组。
    处理colspan和rowspan。
    """
    soup = BeautifulSoup(html_table, 'html.parser')
    table_body = soup.find('tbody') if soup.find('tbody') else soup
    rows = table_body.find_all('tr')
    cols = max([len(row.find_all(['td', 'th'])) for row in rows])
    table = [[None] * cols for _ in rows]
    
    for row in rows:
        col = 0
        for cell in row.find_all(['td', 'th']):
            rowspan = cell.attrs.get('rowspan', 1)
            colspan = cell.attrs.get('colspan', 1)
            print(row.index)
            table[row.index][col:col+colspan] = [cell.text] + [None] * (colspan - 1)
            for extra_row in range(rowspan - 1):
                if len(table) <= row.index + extra_row + 1:
                    table.append([None] * cols)
                table[row.index + extra_row + 1][col:col+colspan] = [None] * colspan
            col += colspan
    
    return table

def change_html_table_to_md(html):
    soup = BeautifulSoup(html, 'html.parser')
    md_table = ''
    
    for row in soup.find('table').find_all('tr'):
        md_row = '| '
        for cell in row.find_all(['td', 'th']):
            md_row += cell.get_text() + ' | '
            
            # Handle colspan
            colspan = cell.get('colspan')
            if colspan:
                md_row += ' ' * (int(colspan) - 1)
        
        # Handle rowspan
        rowspan = cell.get('rowspan')
        if rowspan and int(rowspan) > 1:
            md_row += '\n'
            for _ in range(1, int(rowspan)):
                md_row += '| ' + ' ' * len(cell.get_text()) + ' | ' + '\n'
        
        md_table += md_row + '\n'
    
    # Add header separator
    md_table = md_table.replace('|---', '| --- ')
    return md_table

# def fillin_prompt_template(pt):
# def change_html_table_to_md(html_table):
    
    # 使用BeautifulSoup解析HTML表格
    soup = BeautifulSoup(html_table, 'html.parser')
    
    # 转换为Markdown格式
    md_table = ""
    
    # 添加表格行
    for row in soup.find_all('tr'):
        md_row = "| "
        for cell in row.find_all(['th', 'td']):
            md_row += cell.get_text() + " | "
        md_table += md_row + "\n"
    
    # 添加表格分隔线
    header_row = soup.find('tr')
    md_table += "| " + " | ".join(['---' for cell in header_row.find_all(['th'])]) + " |\n"
    
    print(md_table)
    return md_table


def get_prompt_template(fn):
    pf = open(fn, 'r')
    return pf


def get_markdown_table(table, without_header_line=True):
    table_content = ""

    header = table[0]
    new_header = [str(r) if r != "" else "---" for r in header]
    header_str = "|".join(new_header)
    header_str = "|" + header_str + "|\n"

    table_content += header_str

    if without_header_line == False:
        header_md_split = [":---:" for i in range(len(new_header))]
        header_md_split_str = "|".join(header_md_split)
        header_md_split_str = "|" + header_md_split_str + "|\n"
        
        table_content += header_md_split_str

    for j in range(1,len(table)):
        # print("j : ", j)
        row = table[j]
        new_row = [str(r) if r != "" else "---" for r in row]
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


def concat_table_description(table_description):
    tb = ""
    for key, value in table_description.items():
        tb += value + "\n"

    return tb


def concat_evidence(paras, tables, text_evidence, table_evidence):
    table_str_list = []
    para_str_list = []
    if len(table_evidence) > 0:
        for x in table_evidence:
            y = tables[x]
            table_str_list.append(y)

    if len(text_evidence) > 0:
        for x in text_evidence:
            y = paras[x]
            para_str_list.append(y)

    para_final_evi = " ## ".join(para_str_list)
    table_final_evi = " ## ".join(table_str_list)
    l = []
    if para_final_evi != "":
        l.append(para_final_evi)
    l.append(table_final_evi)

    out_str = " ## ".join(l)
    # out_str = table_final_evi
    return out_str


def concat_evidence_table_raw(paras, raw_tables_df_list, text_evidence, table_evidence):
    table_str_list = []
    para_str_list = []
    if len(table_evidence) > 0:
        for x in table_evidence:
            all = x.split("-")
            table_idx, row, col = all[0], all[1], all[2]
            current_table = raw_tables_df_list[int(table_idx)]
            current_cell = current_table[int(row)][int(col)]
            table_str_list.append(str(current_cell))

    if len(text_evidence) > 0:
        for x in text_evidence:
            y = paras[x]
            para_str_list.append(y)

    para_final_evi = " ## ".join(para_str_list)
    table_final_evi = " ## ".join(table_str_list)
    l = []
    if para_final_evi != "":
        l.append(para_final_evi)
    l.append(table_final_evi)

    out_str = " ## ".join(l)
    # out_str = table_final_evi
    return out_str


# "qa": {
#     "question": "Which year is Total Revenues of Group retirement products the most?",
#     "answer": "2006",
#     "table_evidence": [
#         "0-2-4",
#         "0-8-4",
#         "0-14-4"
#     ],
#     "program": "",
#     "text_evidence": [
#         0
#     ],
#     "question_type": "span_selection"
# }


prompt_prefix_len_list = []
response_len_list = []


def construct_the_final_prompt(fn_header, fn_body, fn_dataset):
    pf_header = get_prompt_template(fn_header)
    pf_body = get_prompt_template(fn_body)

    prompt_template = pf_body.read()
    prompt_header = pf_header.read()

    final_prompt_list = []


    f = open(fn_dataset, 'r')
    # f = open('llm/dataset/mht/dev.json','r')

    l = json.loads(f.read())

    i = 0

    final_list = []
    max_length = 0
    for d in l:
        i += 1
        print("--------------")
        quid = d['uid']
        raw_tables = d['tables']
        paras = d['paragraphs']
        table_description = d['table_description']

        qa = d['qa']
        question = qa['question']
        answer = qa['answer']
        program = qa['program']
        question_type = qa['question_type']

        # 问题1. 原始数据集中的输入evidence是对表格单元格数据的一句话描述，这里构造prompt时，evidence的部分是给出这句话，还是具体的数值？
        # 问题2. 训练时，是应该把top30的数据输入训练，还是把原始数据集中的输入训练？如果输入top30，那么ground truth中就有不在top30的情况，这个可能对模型造成困扰。如果用原始数据集中的数据训练，导致输入文本太长
        model_input_text = d['model_input_text']
        # 表格的文本表示形式
        model_input_text_str = "\n".join(model_input_text)


        table_evidence = qa['table_evidence']
        text_evidence = qa['text_evidence']


        question_type = qa['question_type']

        paras_str = " ".join(paras)
        # 原始数据集中的最大量的表格文本
        tb_str = concat_table_description(table_description)

        prompt_body_str = prompt_template

        
        prompt_body_str = prompt_body_str.replace("{question}", question)
        prompt_body_str = prompt_body_str.replace("{text}", paras_str)




        table_raw_array_2d_list = []

        # 处理表格的部分，需要调整 prompt 模板，然后把几个表依次填入
        # 然后 evidence 的地方也要重新构造，把表格中获取的 evidence 填入
        # 原始 html 格式的表格信息
        table_raw_md_all_str = ""
        for table_idx, table in enumerate(raw_tables):
            table_seq_str = "## Table " + str(table_idx) + ":\n"

            table_md = to_df(table)
            table_md = table_md.fillna("")
            # tm = np.array(table_md)
            # print(tm)
            table_array_2d = table_md.to_numpy()
            table_raw_array_2d_list.append(table_array_2d)

            # 获取表格的md形式
            md_table = get_markdown_table(table_array_2d)
            current_table_all_str = table_seq_str + md_table + "\n"

            table_raw_md_all_str += current_table_all_str
        
        


        # 取top30的表格文本部分输入
        # prompt_body_str = prompt_body_str.replace("{table}", model_input_text_str)

        # 把最原始的表格的markdown格式传进去
        prompt_body_str = prompt_body_str.replace("{table}", table_raw_md_all_str)


        # ---------------- 构造证据的部分 -------------------
        # 把 text 和 table的evidence拼接成一整个string
        # evidence_str = concat_evidence(paras, table_description, text_evidence, table_evidence)
        # 把表格最原始的格式传入返回证据
        evidence_str = concat_evidence_table_raw(paras, table_raw_array_2d_list, text_evidence, table_evidence)



        program = "N.A." if question_type == "span_selection" else program
        prompt_body_str = prompt_body_str.replace("{gold_program}", program)
        prompt_body_str = prompt_body_str.replace("{gold_evidence}", evidence_str)
        prompt_body_str = prompt_body_str.replace("{gold_question_type}", question_type)
        prompt_body_str = prompt_body_str.replace("{gold_answer}", str(answer))


        


        final_prompt = prompt_header + "\n" + prompt_body_str

        # 处理多个 ### Response 的情况
        response_index_list = find_all_indexes("### Response", final_prompt)
        the_last_response_index = response_index_list[len(response_index_list)-1]

        total_response_list = final_prompt.split("### Response")

        prompt_prefix = ""

        for reponse_part in total_response_list[:len(total_response_list)-1]:
            prompt_prefix += reponse_part + "### Response"
    
        response_str = final_prompt[len("### Response") + the_last_response_index:]

        prompt_prefix_len_list.append(len(prompt_prefix.split(" ")))
        response_len_list.append(len(response_str.split(" ")))

        # response_index = final_prompt.find("### Response")
        # prompt_prefix = final_prompt.split("### Response")[0] + "### Response"
        # response_str = final_prompt[len("### Response") + response_index:]

        
        print(prompt_prefix)
        # print(final_prompt)
        # print(response_str)
        # exit(0)

        dout = {"quid":quid, "prompt_full": final_prompt, "prompt_prefix":prompt_prefix, "program": program, "answer": str(answer), "response": response_str.replace("####", "")}
        # print(final_prompt)
        final_prompt_list.append(dout)

        # print(context_str)
        # out_d = {"category":"closed_qa", "context": context_str, "instruction": instru, "response": ans}
        # out_str = json.dumps(out_d)
        # fout.write(out_str + "\n")


    print(i)
    print(max_length)
    # print(sum(final_list)/len(final_list))

    return final_prompt_list


def find_all_indexes(pattern, string):
    # 使用re.finditer获取所有匹配的迭代器
    matches = re.finditer(pattern, string)
    # 将匹配的span转换为开始索引并收集
    return [match.span()[0] for match in matches]

# cmd:
# python3 llm/data_process_mht_table_raw.py dev llm/prompts/table_raw/example1


if __name__ == "__main__":

    mode = sys.argv[1]
    parent_dir = sys.argv[2]

    # parent_dir = llm/prompts/table_raw

    fout = open(f'mht_dataset_table_raw_prompt_{mode}_top30.jsonl','w')

    final_prompt_list = construct_the_final_prompt(f"{parent_dir}/mht_prompt_header.txt", 
                               f"{parent_dir}/mht_prompt_body.txt",
                               f'llm/dataset/mht/combine_reason_input_{mode}_top30.json')
    

    max_prompt_len = max(prompt_prefix_len_list)
    avg_prompt_len = sum(prompt_prefix_len_list) / len(prompt_prefix_len_list)

    max_response_len = max(response_len_list)
    avg_response_len = sum(response_len_list) / len(response_len_list)


    print(f"max_prompt_len : {max_prompt_len}")
    print(f"avg_prompt_len : {avg_prompt_len}")
    print(f"max_response_len : {max_response_len}")
    print(f"avg_response_len : {avg_response_len}")
    

    for dout in final_prompt_list:
        fout.write(json.dumps(dout) + "\n")
    
    fout.flush()
    fout.close()
    