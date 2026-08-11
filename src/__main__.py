from .parser import Parser
from .selector import Selector
from .utils import dump_result
from sys import argv
from pydantic import ValidationError

if __name__ == "__main__":
    try:
        print(argv)
        paths = Parser.parse_cli_args(argv)
        print(paths)
        func_file = paths["--functions_definition"]
        prompt_file = paths["--input"]
        output_file = paths["--output"]
        prompts = Parser.parse_prompt_file(prompt_file)
        try:
            function_definitions = Parser.parse_func_file(func_file)
        except ValidationError as e:
            print(e)
            exit(1)
        s = Selector(prompts, function_definitions)
        result = s.generate_answers()
        dump_result(output_file, result)
    except BaseException as e:
        print()
