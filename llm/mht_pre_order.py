import re

class Stack:
    def __init__(self):
        self.items = []
    def is_empty(self):
        return self.items == []
 
    def push(self, item):
        self.items.append(item)
 
    def pop(self):
        return self.items.pop()

    def pop_n(self, n):
        return self.items.pop(n)

    def peek(self):
        return self.items[len(self.items) - 1]
    
    def size(self):
        return len(self.items)
 




# 把形如  add(10881,8729), divide(#0,const_2) 转换为 --->   divide add 10881 8729 const_2
def infix_to_prefix(infix_expr):
    prec = dict()
    prec[")"] = 4
    prec["*"] = 3
    prec["/"] = 3
    prec["+"] = 2
    prec["-"] = 2
    prec["("] = 1
    prefix_expr = []
    s = Stack()
    for item in reversed(infix_expr.split()):
        if item not in prec.keys():
            prefix_expr.append(item)
        elif item == ')':
            s.push(item)
        elif item == '(':
            while s.peek() != ')':
                prefix_expr.append(s.pop())
            s.pop()
        else:
            while (not s.is_empty())\
                    and s.peek() != ')'\
                    and prec[s.peek()] > prec[item]:
                prefix_expr.append(s.pop())
                s.push(item)
            s.push(item)
    while not s.is_empty():
        prefix_expr.append(s.pop())
    prefix_expr.reverse()
    return ' '.join(prefix_expr)
 
 


# 对前序遍历的表达式进行求值的操作
def mht_program_expr_eval(prefix_expr):
    s = Stack()
    for item in reversed(prefix_expr.split()):
        if item not in OPERATOR_MAPPING:
            s.push(item)
        else:
            op1_str = str(s.pop())
            op2_str = str(s.pop())
            
            if op1_str.lower().startswith("const_"):
                op1_str = op1_str.replace("const_","")

            if op2_str.lower().startswith("const_"):
                op2_str = op2_str.replace("const_","")

            op1 = float(op1_str)
            op2 = float(op2_str)
            result = do_match(item, op1, op2)
            s.push(result)
    return result
 


def do_match(op, op1, op2):
    if op == 'add':
        return op1 + op2
    elif op == 'subtract':
        return op1 - op2
    elif op == 'multiply':
        return op1 * op2
    elif op == 'divide':
        return op1 / op2
    elif op == 'exp':
        return op1 ** op2
    else:
        raise Exception('Error operation!')
    


OPERATOR_MAPPING = {
    "add":"+",
    "subtract":"-",
    "multiply":"*",
    "divide":"/",
    "exp":"**"
}


class OpTreeNode():
    def __init__(self, operation) -> None:
        self.operation = operation
        self.left_node = None
        self.right_node = None
        self.pre_order_list = []

        
def pre_order(root, pre_order_list):
    # 先序遍历
    pre_order_list.append(root.operation.cmd)

    if root.left_node is not None and root.right_node is None:
        pre_order(root.left_node, pre_order_list)
        pre_order_list.append(root.operation.op2)
        # print(root.operation)
    elif root.right_node is not None and root.left_node is None:
        pre_order(root.right_node, pre_order_list)
        pre_order_list.append(root.operation.op1)
        # print(root.operation)
    elif root.left_node is not None and root.right_node is not None:
        pre_order(root.left_node, pre_order_list)
        pre_order(root.right_node, pre_order_list)
        # print(root.operation)
    else:
        # print(root.operation)
        pre_order_list.append(root.operation.op1)
        pre_order_list.append(root.operation.op2)


def do_pre_order_total_list(root):
    pre_order_list = []
    pre_order(root, pre_order_list)
    return pre_order_list



def traverse(root):

    if root.left_node is not None and root.right_node is None:
        traverse(root.left_node)
        print(root.operation)
    elif root.right_node is not None and root.left_node is None:
        traverse(root.right_node)
        print(root.operation)
    elif root.left_node is not None and root.right_node is not None:

        # 判断是否需要调整输出顺序的
        # if root.operation.op1.startswith("#") \
        #     and root.operation.op2.startswith("#"):
        #     # 这种情况就是根据数组中的定位来连接
        #     pattern = re.compile("#(\d+)")

        #     op1_groups = pattern.search(root.operation.op1)
        #     op1_location = int(op1_groups.groups()[0])


        #     op2_groups = pattern.search(root.operation.op2)
        #     op2_location = int(op2_groups.groups()[0])

        #     if op1_location > op2_location:
        #         # 如果左边的location序号比右边的大, 就先输出右子树，再输出左子树
        #         traverse(root.right_node)
        #         traverse(root.left_node)
        #         print(root.operation)
        #     else:
        #         traverse(root.left_node)
        #         traverse(root.right_node)
        #         print(root.operation)
        # else:
        #     traverse(root.left_node)
        #     traverse(root.right_node)
        #     print(root.operation)
        
        traverse(root.left_node)
        traverse(root.right_node)
        print(root.operation)

    else:
        print(root.operation)


# def traverse(root, have_output):

#     if root.left_node is not None and root.right_node is None:
#         traverse(root.left_node, have_output)
#         if str(root.operation) not in have_output:
#             print(root.operation)
#             have_output.append(str(root.operation))
#     elif root.right_node is not None and root.left_node is None:
#         traverse(root.right_node, have_output)
#         if str(root.operation) not in have_output:
#             print(root.operation)
#             have_output.append(str(root.operation))
#     elif root.left_node is not None and root.right_node is not None:
#         traverse(root.left_node, have_output)
#         traverse(root.right_node, have_output)
#         if str(root.operation) not in have_output:
#             print(root.operation)
#             have_output.append(str(root.operation))
#     else:
#         if str(root.operation) not in have_output:
#             print(root.operation)
#             have_output.append(str(root.operation))


class Operation():
    def __init__(self, cmd, op1, op2) -> None:
        self.cmd = cmd
        self.op1 = op1
        self.op2 = op2


    def execute():
        pass


    def __str__(self) -> str:
        return f"{self.cmd}[{self.op1},{self.op2}]"




def parse_program_str(program_str, to_op=False):
    state = 'INIT'
    tmp = ""
    current_cmd = ""

    operation_list = []
    for i in range(len(program_str)):
        ch = program_str[i]

        if ch == '(' and state == "INIT":
            state = "OP_START"
            current_cmd = tmp
            # 清空 tmp
            tmp = ""
        elif ch == ')' and state == "OP_START":
            state = "OP_END"
            tmp_op_list = tmp.split(",")
            op1 = tmp_op_list[0]
            op2 = tmp_op_list[1]

            if to_op is True:
                current_cmd = OPERATOR_MAPPING[current_cmd.strip()]

            operation = Operation(current_cmd.strip(), op1, op2)
            operation_list.append(operation)
            
            current_cmd = ""
            tmp = ""
        elif (ch == "," or ch == " ") and state == "OP_END":
            state = "INIT"
            continue
        else:
            tmp += ch

    # print("##############")
    # for oper in operation_list:
    #     print(oper)

    # parse_to_optree(operation_list)
    pre_order_list = parse_to_optree_without_stack(operation_list)
    # print("##############")
    return pre_order_list
    



def parse_to_optree(operation_list):
    # 如果遇到 # 号，就是要把前面一个操作挂接在这个位置，需要一个stack
    tree_stack = Stack()

    tree_node_list = [OpTreeNode(op) for op in operation_list]

    root = tree_node_list[len(tree_node_list)-1]

    for idx, tree_node in enumerate(tree_node_list):
        if tree_node.operation.op1.startswith("#") and not tree_node.operation.op2.startswith("#"):
            
            # 如果现在stack里面存储的数量大于1， 说明有前置分离的叶子节点，需要处理
            # if tree_stack.size() >= 2:
            #     # pattern = re.compile("#(\d+)")
            #     # op1_groups = pattern.search(tree_node.operation.op1)
            #     # op1_location = int(op1_groups.groups()[0])

            #     # 如果stack中的元素大于1，那么#是几就取几号元素
            #     # if op1_location == 0:
            #     #     left_node = tree_stack.pop_n(0)
            #     # elif op1_location == 1:
            #     #     left_node = tree_stack.pop()

            #     left_node = tree_stack.pop_n(0)
            #     tree_node.left_node = left_node
                
            # else:
            #     left_node = tree_stack.pop()
            #     tree_node.left_node = left_node
            
            left_node = tree_stack.pop()
            tree_node.left_node = left_node
            
            tree_stack.push(tree_node)
        elif tree_node.operation.op2.startswith("#") and not tree_node.operation.op1.startswith("#"):
            
            # if tree_stack.size() >= 2:
            #     # pattern = re.compile("#(\d+)")
            #     # op2_groups = pattern.search(tree_node.operation.op2)
            #     # op2_location = int(op2_groups.groups()[0])
            
                
            #     # 如果stack中的元素大于1，那么#是几就取几号元素
            #     # if op2_location == 0:
            #     #     right_node = tree_stack.pop_n(0)
            #     # elif op2_location == 1:
            #     #     right_node = tree_stack.pop()

            #     right_node = tree_stack.pop_n(0)
            #     tree_node.right_node = right_node

            # else:
            #     right_node = tree_stack.pop()
            #     tree_node.right_node = right_node

            right_node = tree_stack.pop()
            tree_node.right_node = right_node
            
            tree_stack.push(tree_node)
        elif tree_node.operation.op1.startswith("#") and tree_node.operation.op2.startswith("#"):
            # 这种情况就是根据数组中的定位来连接
            pattern = re.compile("#(\d+)")

            op1_groups = pattern.search(tree_node.operation.op1)
            op1_location = int(op1_groups.groups()[0])
            tree_node.left_node = tree_node_list[op1_location]


            op2_groups = pattern.search(tree_node.operation.op2)
            op2_location = int(op2_groups.groups()[0])
            tree_node.right_node = tree_node_list[op2_location]



            # op1_sharp_idx = tree_node.operation.op1.find("#")
            # op1_location = int(tree_node.operation.op1[op1_sharp_idx+1])
            # tree_node.left_node = tree_node_list[op1_location]

            # op2_sharp_idx = tree_node.operation.op2.find("#")
            # op2_location = int(tree_node.operation.op2[op2_sharp_idx+1])
            # tree_node.right_node = tree_node_list[op2_location]

            tree_stack.push(tree_node)
        else:
            # print("push1")
            tree_stack.push(tree_node)

    have_output = []
    # traverse(root, have_output)
    # traverse(root)
    pre_order_list = do_pre_order_total_list(root)
    print(pre_order_list)



def parse_to_optree_without_stack(operation_list):
    # 如果遇到 # 号，就是要把前面一个操作挂接在这个位置，需要一个stack
    # tree_stack = Stack()

    tree_node_list = [OpTreeNode(op) for op in operation_list]

    root = tree_node_list[len(tree_node_list)-1]

    for idx, tree_node in enumerate(tree_node_list):
        if tree_node.operation.op1.startswith("#") and not tree_node.operation.op2.startswith("#"):
            
            # 如果现在stack里面存储的数量大于1， 说明有前置分离的叶子节点，需要处理
            pattern = re.compile("#(\d+)")
            op1_groups = pattern.search(tree_node.operation.op1)
            op1_location = int(op1_groups.groups()[0])

            tree_node.left_node = tree_node_list[op1_location]
            
        elif tree_node.operation.op2.startswith("#") and not tree_node.operation.op1.startswith("#"):
            
            pattern = re.compile("#(\d+)")
            op2_groups = pattern.search(tree_node.operation.op2)
            op2_location = int(op2_groups.groups()[0])
            
            tree_node.right_node = tree_node_list[op2_location]
                
                # 如果stack中的元素大于1，那么#是几就取几号元素
        elif tree_node.operation.op1.startswith("#") and tree_node.operation.op2.startswith("#"):
            # 这种情况就是根据数组中的定位来连接
            pattern = re.compile("#(\d+)")

            op1_groups = pattern.search(tree_node.operation.op1)
            op1_location = int(op1_groups.groups()[0])
            tree_node.left_node = tree_node_list[op1_location]


            op2_groups = pattern.search(tree_node.operation.op2)
            op2_location = int(op2_groups.groups()[0])
            tree_node.right_node = tree_node_list[op2_location]

        else:
            pass

    # traverse(root, have_output)
    # traverse(root)

    pre_order_list = do_pre_order_total_list(root)
    # print(pre_order_list)
    return pre_order_list


if __name__ == "__main__":
    # fin = open("program_list.txt","r")
    # for line in fin:
    #     parse_program_str(line)




    # line = "add(3404,3110), add(#0,3122), add(2624,51579), add(#2,11657), add(#3,618), add(#4,3404), add(#5,3110), add(#6,3122), divide(#1,#7)"
    # line = "add(91,68), add(#0,3), add(#1,20), add(#2,10.11), add(#3,8.24), add(91,68), add(#5,3), add(#6,20), add(#7,10.11), add(#8,8.24), add(#9,159), add(#10,162), add(#11,182), divide(#4,#12)"
    # parse_program_str(line)


    # line = "subtract(116.62,100),subtract(107.65,100),subtract(105.45,100),add(#0,#1),add(#2,#3)"

    line = "divide(115.18,100),divide(113.68,100),subtract(#0,const_1),subtract(#1,const_1),add(#2,#3),divide(#4,const_2)"
    
    # line = "subtract(20.9,20.2),divide(#0,20.2),subtract(21.3,20.9),divide(#2,20.9)"
    


    # line = "multiply(400,3.25%),multiply(400,2.00%),add(#1,#0)"
    # line = "add(450000,500000), add(#0,625000), add(#0,#1), add(#2,300000)"

    l = parse_program_str(line)
    prog_prefix_expr = " ".join(l)
    print(prog_prefix_expr)

    ans = mht_program_expr_eval(prog_prefix_expr)
    print(ans)

    

