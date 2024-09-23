import re



ss = '''| 4 | 4.70588 |'''
ss = '''| 4 | 4.70588asdf |'''
ss = '''| 4 | -2.';p[4.70588asdf |'''

ss4 = '''or more late. (b) Vintage refers to the age of the account.\n\n### Question\nWhat is the percentage change in the number of accounts re-defaulted from 2011 to 2012 for the residential mortgage servicing rights?\n\n### Response\n| step | output |\n| 1 | arithmetic |\n| 2 | 42 ## 238 |\n| 3 | subtract(42,238), divide(#0,238) |\n| 4 | -0.83333 |\nThe program is'''

sxx = '''| 4 | -0.83333 |'''

sss = '''| 3 | subtract(2453,2750), divide(#0,2750) |'''

ss_real = '''ents as of September 30, 2008):

### Question
What is the percentage change in the total cash and cash equivalents from 2009 to 2010?

### Response
| step | output |
| 1 | arithmetic |
| 2 | 2943239 ## 2306085 |
| 3 | subtract(2943239,2306085), divide(#0,2306085) |
| 4 |'''



ss_prog_ans = '''
\n | step | output |\n| 1 | arithmetic |\n| 2 | 475|$1,250|\n| 3 | subtract(475,1,250), divide(#0,1,250) |\n| 4 | 0.376|\n The program is: <Prog>subtract(475,1,250), divide(#0,1,250)</Prog>, and the answer is: <Ans>0.376</Ans>
'''


reg_prog = r'.*<Prog>(.*)</Prog>.*'
reg_ans = r'.*<Ans>(.*)</Ans>.*'






reg3 = r'.*\|.*3.*\|(.*)\|.*'
reg4 = r'.*\|\s4\s\|(.*)\|.*'

p4 = re.compile(reg4)
p3 = re.compile(reg3)



p_prog = re.compile(reg_prog)
p_ans = re.compile(reg_ans)


m4 = p4.search(ss4)
if m4:
    print("m4")
    print(m4.group(1))


m3 = p3.search(ss_real)
if m3:
    print("m3")
    print(m3.group(1))


m_ans = p_ans.search(ss_prog_ans)
if m_ans:
    print("m_ans")
    print(m_ans.group(1))



