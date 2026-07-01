import sys
from parser import Parser

if __name__ == '__main__':
    if len(sys.argv) == 1:
        input_file = ""
    prompts = Parser.parse_input_file('../data/input/function_calling_tests.json')
    functions = Parser.parse_func_file('../data/input/functions_definition.json')
    # d = {'a':1}
    for k in prompts:
        print(k.prompt)

