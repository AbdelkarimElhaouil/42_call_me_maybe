from sys import exit
from pydantic import BaseModel
import json

class FunctionDefinition:
    def __init__(self, name: str, description: str,
                 parameters: dict, returns: dict) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.returns = returns
        

class Parser:
    # def __init__(self, path:str):
    #     self.path = path
    def parse_func_file(path: str) -> list[FunctionDefinition]:
        try:
            with open(path) as file:
                json.load
                pass

        except IOError as e:
            print(f"Error occured while opening file {path}: {e}")
            exit(1)

    def parse_input_file(path: str) -> list[str]:
        pass 