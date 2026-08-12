"""Run the main workflow of the function-calling application."""

from .parser import Parser
from .selector import Selector
from .utils import dump_result
from sys import argv

if __name__ == "__main__":
    try:
        paths = Parser.parse_cli_args(argv)
        func_file = paths["--functions_definition"]
        prompt_file = paths["--input"]
        output_file = paths["--output"]
        prompts = Parser.parse_prompt_file(prompt_file)
        function_definitions = Parser.parse_func_file(func_file)
        s = Selector(prompts, function_definitions)
        result = s.generate_answers()
        dump_result(output_file, result)
    except BaseException as e:
        print("hchem chwia")
