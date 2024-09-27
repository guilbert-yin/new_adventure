import json, os, re, sys
from mht_utils.eval_utils import *
import math
import mht_pre_order as mht_pre_order

def str_to_num(text):
    text = text.replace("$","")
    text = text.replace(",", "")
    text = text.replace("-", "")
    text = text.replace("%", "")
    try:
        num = float(text)
    except ValueError:
        if "const_" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            num = float(text)
        else:
            num = "n/a"
    return num


def evaluate_span_program_result(span_ans, prog_ans):
    span_ans = str(span_ans)
    if str_to_num(span_ans) != "n/a":
        span_ans = str_to_num(span_ans)
        if math.isclose(prog_ans, span_ans, abs_tol= min(abs(min(prog_ans, span_ans) / 1000), 0.1)):
            exact_match, f1 = 1, 1
        else:
            exact_match, f1 = 0, 0
    else:
        exact_match, f1 = get_span_selection_metrics(span_ans, str(prog_ans))
    return exact_match, f1



def evaluation_prediction_result_reverse(predict_json_in, gold_json_in, test_file_json_in, output_dir):
    exact_match_total, f1_total = 0, 0

    prediction_dict = {}

    predict_json_in_f = open(predict_json_in, "r")

    num_examples = 0

    orig_data = json.load(open(gold_json_in))
    orig_data_dict = {}
    for example in orig_data:
        uid = example["uid"]
        orig_data_dict[uid] = example


    span_em = []
    span_f1 = []

    arith_em = []
    arith_f1 = []


    for line in predict_json_in_f:
        js_l = json.loads(line)
        response = js_l['res']

        # clean data
        # response = response.replace(": Answer:","")
        # response = response.replace(": The answer is:","")
        # response = response.replace("## Answer:","")
        # response = response.replace("\n The answer is:","")
        # response = response.replace("The answer is:","")
        # response = response.replace(":","")

        response = response_cleaner_strategy(response)

        quid = js_l['quid']

        prediction_dict[js_l['quid']] = str(response)
        pred = str(response)

        num_examples += 1

        example = orig_data_dict[quid]

        gold_prog = example["qa"]["program"]
        gold_ans = example["qa"]["answer"]
        question_type = example["qa"]["question_type"]




        # both span selection
        exact_acc, f1_acc = get_span_selection_metrics(pred, str(gold_ans))

        if question_type == "span_selection":
            span_em.append(exact_acc)
            span_f1.append(f1_acc)
        
        elif question_type == "arithmetic":
            arith_em.append(exact_acc)
            arith_f1.append(f1_acc)

        # gold is span selection, pred is program generation
        # elif not pred["predicted_program"] and gold_prog:
        #     exact_acc, f1_acc = evaluate_span_program_result(span_ans = pred["predicted_ans"], prog_ans = gold_ans)
        # # gold is program generation, pred is span selection
        # elif pred["predicted_program"] and not gold_prog:
        #     exact_acc, f1_acc = evaluate_span_program_result(span_ans = gold_ans, prog_ans = pred["predicted_ans"])

        exact_match_total += exact_acc
        f1_total += f1_acc

    




    exact_match_score, f1_score = exact_match_total / num_examples, f1_total / num_examples
    print(f"Total Exact Match Score: {exact_match_score}, F1 Score: {f1_score}")
    print("================================")

    span_em_overall = sum(span_em) / len(span_em)
    span_f1_overall = sum(span_f1) / len(span_f1)

    span_em_intotal = sum(span_em) / num_examples
    span_f1_intotal = sum(span_f1) / num_examples

    arith_em_overall = sum(arith_em) / len(arith_em)
    arith_f1_overall = sum(arith_f1) / len(arith_f1)

    arith_em_intotal = sum(arith_em) / num_examples
    arith_f1_intotal = sum(arith_f1) / num_examples

    print(f"Span Exact Match Score: {span_em_overall}, F1 Score: {span_f1_overall}")
    print(f"Span In Total Exact Match Score: {span_em_intotal}, F1 Score: {span_f1_intotal}")
    print("---------------------------------")
    print(f"Arithmetic Exact Match Score: {arith_em_overall}, F1 Score: {arith_f1_overall}")
    print(f"Arithmetic In Total Exact Match Score: {arith_em_intotal}, F1 Score: {arith_f1_intotal}")

    return exact_match_score, f1_score


all_ops = ["add", "subtract", "multiply", "divide", "exp"]


def eval_program(program):
    '''
    calculate the numerical results of the program
    '''

    invalid_flag = 0
    this_res = "n/a"

    try:
        # program = program[:-1]  # remove EOF
        # check structure
        for ind, token in enumerate(program):
            if ind % 4 == 0:
                if token.strip("(") not in all_ops:
                    return 1, "n/a"
            if (ind + 1) % 4 == 0:
                if token != ")":
                    return 1, "n/a"

        program = "|".join(program)
        steps = program.split(")")[:-1]

        res_dict = {}

        for ind, step in enumerate(steps):
            step = step.strip()

            if len(step.split("(")) > 2:
                invalid_flag = 1
                break
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip()

            arg1 = args.split("|")[0].strip()
            arg2 = args.split("|")[1].strip()

            if "#" in arg1:
                arg1 = res_dict[int(arg1.replace("#", ""))]
            else:
                arg1 = str_to_num(arg1)
                if arg1 == "n/a":
                    invalid_flag = 1
                    break

            if "#" in arg2:
                arg2 = res_dict[int(arg2.replace("#", ""))]
            else:
                arg2 = str_to_num(arg2)
                if arg2 == "n/a":
                    invalid_flag = 1
                    break

            if op == "add":
                this_res = arg1 + arg2
            elif op == "subtract":
                this_res = arg1 - arg2
            elif op == "multiply":
                this_res = arg1 * arg2
            elif op == "divide":
                this_res = arg1 / arg2
            elif op == "exp":
                this_res = arg1 ** arg2

            res_dict[ind] = this_res

        if this_res != "n/a":
            this_res = round(this_res, 5)

    except:
        invalid_flag = 1

    return invalid_flag, this_res


# def evaluate_program_result(pred_prog, gold_prog):
#     '''
#     execution acc
#     execution acc = exact match = f1
#     '''
#     invalid_flag, exe_res = eval_program(pred_prog)

#     gold = program_tokenization(gold_prog)
#     invalid_flag, exe_gold_res = eval_program(gold)

#     if invalid_flag:
#         print(gold)
#     if exe_res == exe_gold_res:
#         exe_acc = 1
#     else:
#         exe_acc = 0

#     return exe_acc, exe_acc


# 形式 1:
# deferred tax assets and liabilities are not disclosed in the financial statements.
### Question
# What is the percentage change in the unrecognized tax benefits from 2015 to 2016?
# ### Response
# | step | output |
# | 1 | arithmetic |
# | 2 | 88 ## 17 |
# | 3 | subtract(88,17), divide(#0,17) |
# | 4 | 4.70588 |
# The program is: <Prog>subtract(88,17),


# 形式 2:
# operating expenses and unexpected increases in the cost of capital could result in a significant reduction in the Corporation\u2019s cash flow and earnings.
### Question
# What is the percentage change in the sales and other operating revenues from 2010 to 2011?
# ### Response
# | step | output |
# | 1 | arithmetic |
# | 2 | 2453 ## 2750 |
# | 3 | subtract(2453,2750), divide(#0,2750) |
# | 4 | -0.1



def response_cleaner_strategy(response):
    reg_prog = r'.*<Prog>(.*)</Prog>.*'
    reg_ans = r'.*<Ans>(.*)</Ans>.*'

    p_prog = re.compile(reg_prog)
    p_ans = re.compile(reg_ans)

    # 先判断如果p4命中则直接比对
    m_ans = p_ans.search(response)
    if m_ans:
        res = m_ans.group(1)
        print("---------")
        print(res)
        # print(response)
        print("=========")
        return res
    

    # print("prepare m3")
    # print(response)
    
    m_prog = p_prog.search(response)
    if m_prog:
        program = m_prog.group(1)
        # print(program)
        # 执行程序
        try:
            l = mht_pre_order.parse_program_str(program)
            prog_prefix_expr = " ".join(l)
            print(prog_prefix_expr)

            ans = mht_pre_order.mht_program_expr_eval(prog_prefix_expr)
            return ans
        except:
            return response
    
        
    return response





def evaluation_prediction_result(predict_json_in, gold_json_in, test_file_json_in, output_dir):
    exact_match_total, f1_total = 0, 0

    prediction_dict = {}

    predict_json_in_f = open(predict_json_in, "r")
    for line in predict_json_in_f:
        js_l = json.loads(line)
        response = js_l['res']

        # clean data
        response = response.replace(": Answer:","")
        response = response.replace(": The answer is:","")
        response = response.replace("## Answer:","")
        response = response.replace("\n The answer is:","")
        response = response.replace("The answer is:","")
        response = response.replace(":","")

        response = response_cleaner_strategy(response)

        prediction_dict[js_l['quid']] = str(response)


    orig_data = json.load(open(gold_json_in))

    num_examples = len(orig_data)

    span_em = []
    span_f1 = []

    arith_em = []
    arith_f1 = []

    for example in orig_data:
        uid = example["uid"]
        pred = prediction_dict[uid]

        gold_prog = example["qa"]["program"]
        gold_ans = example["qa"]["answer"]
        question_type = example["qa"]["question_type"]




        # both span selection
        exact_acc, f1_acc = get_span_selection_metrics(pred, str(gold_ans))

        if question_type == "span_selection":
            span_em.append(exact_acc)
            span_f1.append(f1_acc)
        
        elif question_type == "arithmetic":
            arith_em.append(exact_acc)
            arith_f1.append(f1_acc)

        # gold is span selection, pred is program generation
        # elif not pred["predicted_program"] and gold_prog:
        #     exact_acc, f1_acc = evaluate_span_program_result(span_ans = pred["predicted_ans"], prog_ans = gold_ans)
        # # gold is program generation, pred is span selection
        # elif pred["predicted_program"] and not gold_prog:
        #     exact_acc, f1_acc = evaluate_span_program_result(span_ans = gold_ans, prog_ans = pred["predicted_ans"])

        exact_match_total += exact_acc
        f1_total += f1_acc


    exact_match_score, f1_score = exact_match_total / num_examples, f1_total / num_examples
    print(f"Total Exact Match Score: {exact_match_score}, F1 Score: {f1_score}")
    print("================================")

    span_em_overall = sum(span_em) / len(span_em)
    span_f1_overall = sum(span_f1) / len(span_f1)

    span_em_intotal = sum(span_em) / num_examples
    span_f1_intotal = sum(span_f1) / num_examples

    arith_em_overall = sum(arith_em) / len(arith_em)
    arith_f1_overall = sum(arith_f1) / len(arith_f1)

    arith_em_intotal = sum(arith_em) / num_examples
    arith_f1_intotal = sum(arith_f1) / num_examples

    print(f"Span Exact Match Score: {span_em_overall}, F1 Score: {span_f1_overall}")
    print(f"Span In Total Exact Match Score: {span_em_intotal}, F1 Score: {span_f1_intotal}")
    print("---------------------------------")
    print(f"Arithmetic Exact Match Score: {arith_em_overall}, F1 Score: {arith_f1_overall}")
    print(f"Arithmetic In Total Exact Match Score: {arith_em_intotal}, F1 Score: {arith_f1_intotal}")

    return exact_match_score, f1_score



#执行命令：
# python3 llm/eval_mht.py output/mht_chatglm3_dev_top30_out.jsonl llm/dataset/mht/dev.json

if __name__ == '__main__':
    predict_file = sys.argv[1]
    gold_file = sys.argv[2]
    r = sys.argv[3]

    if r == "r":
        evaluation_prediction_result_reverse(predict_file, gold_file, None, None)
    else:
        evaluation_prediction_result(predict_file, gold_file, None, None)





# 240915 chatglm3 initial
# Total Exact Match Score: 0.0603448275862069, F1 Score: 0.07159961685823754
# ================================
# Span Exact Match Score: 0.29207920792079206, F1 Score: 0.3428217821782179
# Span In Total Exact Match Score: 0.05651340996168582, F1 Score: 0.06633141762452109
# ---------------------------------
# Arithmetic Exact Match Score: 0.004750593824228029, F1 Score: 0.006532066508313539
# Arithmetic In Total Exact Match Score: 0.0038314176245210726, F1 Score: 0.005268199233716475




# 240924 chatglm3 refined with example table str evidence str
# Total Exact Match Score: 0.027777777777777776, F1 Score: 0.035507662835249046
# ================================
# Span Exact Match Score: 0.09900990099009901, F1 Score: 0.13782178217821783
# Span In Total Exact Match Score: 0.019157088122605363, F1 Score: 0.02666666666666667
# ---------------------------------
# Arithmetic Exact Match Score: 0.010688836104513063, F1 Score: 0.010961995249406175
# Arithmetic In Total Exact Match Score: 0.008620689655172414, F1 Score: 0.008840996168582374




# 20240923 chatglm3 refined with example3 table raw evidence str
# 
# Total Exact Match Score: 0.01053639846743295, F1 Score: 0.01553639846743295
# ================================
# Span Exact Match Score: 0.04950495049504951, F1 Score: 0.07262376237623762
# Span In Total Exact Match Score: 0.009578544061302681, F1 Score: 0.014051724137931034
# ---------------------------------
# Arithmetic Exact Match Score: 0.0011876484560570072, F1 Score: 0.0018408551068883612
# Arithmetic In Total Exact Match Score: 0.0009578544061302681, F1 Score: 0.0014846743295019158



# 240924 chatglm3 refined with example4 table raw evidence num
# Total Exact Match Score: 0.016283524904214558, F1 Score: 0.021340996168582375
# ================================
# Span Exact Match Score: 0.04950495049504951, F1 Score: 0.07247524752475247
# Span In Total Exact Match Score: 0.009578544061302681, F1 Score: 0.014022988505747125
# ---------------------------------
# Arithmetic Exact Match Score: 0.00831353919239905, F1 Score: 0.009073634204275535
# Arithmetic In Total Exact Match Score: 0.006704980842911878, F1 Score: 0.007318007662835249





# 240924 chatglm3 refined with example2 table str evidence num
# chatglm3-base_table_str_prompt_example2_evi_num_refine_output_full.jsonl
#
# Total Exact Match Score: 0.022030651340996167, F1 Score: 0.038994252873563236
# ================================
# Span Exact Match Score: 0.08415841584158416, F1 Score: 0.16851485148514858
# Span In Total Exact Match Score: 0.016283524904214558, F1 Score: 0.03260536398467434
# ---------------------------------
# Arithmetic Exact Match Score: 0.007125890736342043, F1 Score: 0.007921615201900238
# Arithmetic In Total Exact Match Score: 0.005747126436781609, F1 Score: 0.006388888888888888





# 20240925 llama2-chat llama2_table_str_prompt_example2_refine_output.jsonl
# Total Exact Match Score: 0.0, F1 Score: 0.0070210727969348685
# ================================
# Span Exact Match Score: 0.0, F1 Score: 0.03371287128712872
# Span In Total Exact Match Score: 0.0, F1 Score: 0.006522988505747129
# ---------------------------------
# Arithmetic Exact Match Score: 0.0, F1 Score: 0.0006175771971496437
# Arithmetic In Total Exact Match Score: 0.0, F1 Score: 0.0004980842911877395





# 20240926 minicpm minicpm_table_raw_prompt_example4_dev_refine_output.jsonl
# Total Exact Match Score: 0.0009578544061302681, F1 Score: 0.0018678160919540236
# ================================
# Span Exact Match Score: 0.0049504950495049506, F1 Score: 0.009554455445544558
# Span In Total Exact Match Score: 0.0009578544061302681, F1 Score: 0.0018486590038314181
# ---------------------------------
# Arithmetic Exact Match Score: 0.0, F1 Score: 2.3752969121140145e-05
# Arithmetic In Total Exact Match Score: 0.0, F1 Score: 1.9157088122605363e-05







# 20240926 minicpm_table_str_prompt_example2_dev_evi_num_refine_output.jsonl
# Total Exact Match Score: 0.012452107279693486, F1 Score: 0.013927203065134099
# ================================
# Span Exact Match Score: 0.019801980198019802, F1 Score: 0.025445544554455444
# Span In Total Exact Match Score: 0.0038314176245210726, F1 Score: 0.004923371647509578
# ---------------------------------
# Arithmetic Exact Match Score: 0.010688836104513063, F1 Score: 0.011163895486935867
# Arithmetic In Total Exact Match Score: 0.008620689655172414, F1 Score: 0.009003831417624522






# equation: 

# 20240926 chatglm3-base_table_str_prompt_equation_dev_evi_num_top30_refine_output_full.jsonl
# Total Exact Match Score: 0.022988505747126436, F1 Score: 0.041264367816091975
# ================================
# Span Exact Match Score: 0.09405940594059406, F1 Score: 0.1827227722772278
# Span In Total Exact Match Score: 0.018199233716475097, F1 Score: 0.03535440613026822
# ---------------------------------
# Arithmetic Exact Match Score: 0.0059382422802850355, F1 Score: 0.007327790973871734
# Arithmetic In Total Exact Match Score: 0.004789272030651341, F1 Score: 0.005909961685823755




# equation hints
# 20240926 chatglm3-base_table_str_prompt_woexample_dev_evi_num_top30_refine_output_full.jsonl
# Total Exact Match Score: 0.03639846743295019, F1 Score: 0.044703065134099614
# ================================
# Span Exact Match Score: 0.15346534653465346, F1 Score: 0.1914851485148515
# Span In Total Exact Match Score: 0.029693486590038315, F1 Score: 0.03704980842911877
# ---------------------------------
# Arithmetic Exact Match Score: 0.00831353919239905, F1 Score: 0.009489311163895488
# Arithmetic In Total Exact Match Score: 0.006704980842911878, F1 Score: 0.0076532567049808434



# minicpm_table_str_prompt_equation_dev_evi_num_top30_refine_output.jsonl
# Total Exact Match Score: 0.013409961685823755, F1 Score: 0.015134099616858238
# ================================
# Span Exact Match Score: 0.0, F1 Score: 0.008811881188118813
# Span In Total Exact Match Score: 0.0, F1 Score: 0.0017049808429118777
# ---------------------------------
# Arithmetic Exact Match Score: 0.0166270783847981, F1 Score: 0.01665083135391924
# Arithmetic In Total Exact Match Score: 0.013409961685823755, F1 Score: 0.01342911877394636




# 20240927 chatglm3-base_table_str_prompt_glm3format_dev_evi_num_top30_refine_output_full.jsonl
# Total Exact Match Score: 0.039272030651340994, F1 Score: 0.045699233716475086
# ================================
# Span Exact Match Score: 0.15841584158415842, F1 Score: 0.19004950495049502
# Span In Total Exact Match Score: 0.03065134099616858, F1 Score: 0.03677203065134099
# ---------------------------------
# Arithmetic Exact Match Score: 0.010688836104513063, F1 Score: 0.011068883610451307
# Arithmetic In Total Exact Match Score: 0.008620689655172414, F1 Score: 0.0089272030651341



# 20240927 chatglm3-base_table_str_prompt_glm3format_woexample_dev_evi_num_top30_refine_output_full.jsonl
# Total Exact Match Score: 0.008620689655172414, F1 Score: 0.03431034482758624
# ================================
# Span Exact Match Score: 0.034653465346534656, F1 Score: 0.1537623762376239
# Span In Total Exact Match Score: 0.006704980842911878, F1 Score: 0.029750957854406155
# ---------------------------------
# Arithmetic Exact Match Score: 0.0023752969121140144, F1 Score: 0.005653206650831353
# Arithmetic In Total Exact Match Score: 0.0019157088122605363, F1 Score: 0.004559386973180077