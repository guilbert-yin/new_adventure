import json, os, re, sys
from mht_utils.eval_utils import *
import math 

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

    evaluation_prediction_result(predict_file, gold_file, None, None)