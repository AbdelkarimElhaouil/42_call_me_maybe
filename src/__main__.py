from .parser import Parser
from .selector import Selector

if __name__ == "__main__":
    prompt_file = "data/input/function_calling_tests.json"
    func_file = "data/input/functions_definition.json"

    prompts = Parser.parse_prompt_file(prompt_file)
    funcs = Parser.parse_func_file(func_file)

    # for p in prompts:
    #     print(p)
    s = Selector(prompts, funcs)

    s.select_func_name()
    