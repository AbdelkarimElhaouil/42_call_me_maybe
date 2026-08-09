from .parser import Parser
from .selector import Selector

if __name__ == "__main__":
    prompt_file = "data/input/function_calling_tests.json"
    func_file = "data/input/functions_definition.json"
    ps = Parser.parse_prompt_file(prompt_file)
    fs = Parser.parse_func_file(func_file)
    s = Selector(ps, fs)
    # s.get_allowed_type_tokens("string")
    s.generate_json()
