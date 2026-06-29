from sys import exit
from pydantic import BaseModel, ValidationError
import json


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]

    @model_validator(mode="after")
    def validate_types(self) -> None:
        allowed_types = [
            "number", "string", "integer",
            "array", "object"
        ]
        for k in self.parameters.keys():
            if self.parameters[k] not in allowed_types:
                print(f"TypeError: {self.parameters[k]} is invalid")
                exit(1)
            if self.returns["type"] not in allowed_types:
                print(f"TypeError: {self.parameters[k]} is invalid")
                exit(1)

class Prompt(BaseModel):
    Prompt: str


class Parser:
    # def __init__(self, path:str):
    #     self.path = path
    def parse_func_file(path: str) -> list[FunctionDefinition]:
        try:
            with open(path) as file:
                data = json.load(file)
                pass

        except IOError as e:
            print(f"Error occured while opening file {path}: {e}")
            exit(1)

    def parse_input_file(path: str) -> list[str]:
        pass