from sys import exit
from pydantic import ValidationError
from models import FunctionDefinition, Prompt
from json import load, JSONDecodeError


class Parser:
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

        except JSONDecodeError as e:
            print(f"Invalid JSON in {path}: {e}")
            exit(1)

    @staticmethod
    def parse_input_file(path: str) -> list[Prompt]:
        prompts: list[Prompt] = []

        try:
            with open(path) as file:
                data =  load(file)
                prompts = [Prompt(**prompt) for prompt in data]
                print(data[2])
                return prompts
        
        except IOError as e:
            print(f"Error occured while opening file {path}: {e}")
            exit(1)

        except (ValidationError, KeyError) as e:
            print(f"Validation error: {e}")
            exit(1)
        
        except JSONDecodeError as e:
            print(f"Invalid JSON in {path}: {e}")
            exit(1)