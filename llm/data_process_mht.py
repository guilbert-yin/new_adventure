import json
# from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import sys

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
        model_input_text_str = "\n".join(model_input_text)


        table_evidence = qa['table_evidence']
        text_evidence = qa['text_evidence']

        # 把 text 和 table的evidence拼接成一整个string
        evidence_str = concat_evidence(paras, table_description, text_evidence, table_evidence)

        question_type = qa['question_type']

        paras_str = " ".join(paras)
        # 原始数据集中的最大量的表格文本
        tb_str = concat_table_description(table_description)

        prompt_body_str = prompt_template

        
        prompt_body_str = prompt_body_str.replace("{question}", question)
        prompt_body_str = prompt_body_str.replace("{text}", paras_str)
        # 取top30的表格文本部分输入
        prompt_body_str = prompt_body_str.replace("{table}", model_input_text_str)

        program = "N.A." if question_type == "span_selection" else program
        prompt_body_str = prompt_body_str.replace("{gold_equation}", program)
        prompt_body_str = prompt_body_str.replace("{gold_evidence}", evidence_str)
        prompt_body_str = prompt_body_str.replace("{gold_question_type}", question_type)
        prompt_body_str = prompt_body_str.replace("{gold_answer}", str(answer))

        # 原始 html 格式的表格信息
        # for table in raw_tables:
        #     table_md = to_df(table)
        #     tm = np.array(table_md)
        #     print(tm)
            # print(table_md.to_numpy())

        final_prompt = prompt_header + "\n" + prompt_body_str

        response_index = final_prompt.find("### Response")
        prompt_prefix = final_prompt.split("### Response")[0] + "### Response"
        response_str = final_prompt[len("### Response") + response_index:]
        print(prompt_prefix)

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

if __name__ == "__main__":

    mode = sys.argv[1]

    fout = open(f'mht_dataset_prompt_{mode}_top30.jsonl','w')
    # pf_header = get_prompt_template("llm/prompts/tat_prompt_header.txt")
    # pf_body = get_prompt_template("llm/prompts/tat_prompt_body.txt")

    final_prompt_list = construct_the_final_prompt("llm/prompts/table_str/mht_prompt_header.txt", 
                               "llm/prompts/table_str/mht_prompt_body.txt",
                               f'llm/dataset/mht/combine_reason_input_{mode}_top30.json')
    
    for dout in final_prompt_list:
        fout.write(json.dumps(dout) + "\n")
    
    fout.flush()
    fout.close()