from sys import exit
from pydantic import ValidationError
from .models import FunctionDefinition, Prompt
from json import load, JSONDecodeError


class Parser:

    @staticmethod
    def parse_cli_args(argv: list[str]) -> dict[str, str]:
        argv_len = len(argv)
        valid_args = ["--input", "--output", "--functions_definition"]
        default_result = {
                "--input": "data/input/function_calling_tests.json",
                "--output": "data/output/function_calling_results.json",
                "--functions_definition": "data/input/functions_definition.json"
        }
        if argv_len > 1:
            i = 1
            while i < argv_len:
                if argv[i] in valid_args and argv_len > i + 1:

                    default_result.update(
                        {argv[i]: argv[i+1]}
                    )
                    i += 2
                else:
                    print("Error: please provide valid command line arguments")
                    print("Usage: uv run python -m src", end="")
                    print("[--functions_definition <function_definition_file>]", end="")
                    print("[--input <input_file>] [--output <output_file>]")
                    exit(1)
        return default_result


    @staticmethod
    def parse_func_file(path: str) -> list[FunctionDefinition]:
        functions: list[FunctionDefinition] = []
        try:
            with open(path) as file:
                data = load(file)
                functions = [FunctionDefinition(**func) for func in data]
                return functions

        except IOError as e:
            print(f"Error occured while opening file {path}: {e}")
            exit(1)

        except ValidationError as e:
            print(f"Validation error: {e}")
            exit(1)
        
        except KeyError as e:
            print(e)
            exit(1)
        
        except ValueError as e:
            print(e)
            exit(1)

        except JSONDecodeError as e:
            print(f"Invalid JSON in {path}: {e}")
            exit(1)

    @staticmethod
    def parse_prompt_file(path: str) -> list[str]:
        prompts: list[Prompt] = []

        try:
            with open(path) as file:
                data =  load(file)
                prompts = [Prompt(**prompt) for prompt in data]
                valid_prompts = [p.prompt for p in prompts if p.prompt.strip() != ""]
                if not valid_prompts:
                    print("All prompts are vide: please provide a valid prompt.")
                    exit(1)
                return valid_prompts
        
        except IOError as e:
            print(f"Error occured while opening file {path}: {e}")
            exit(1)

        except (ValidationError, KeyError) as e:
            print(f"Validation error: {e}")
            exit(1)
        
        except JSONDecodeError as e:
            print(f"Invalid JSON in {path}: {e}")
            exit(1)