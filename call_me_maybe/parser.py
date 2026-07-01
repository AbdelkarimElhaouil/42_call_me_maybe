from sys import exit
from pydantic import BaseModel, ValidationError, model_validator
import json


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]

    @model_validator(mode="after")
    def validate_types(self) -> "FunctionDefinition":
        allowed_types = [
            "number", "string", "integer",
            "array", "object"
        ]
        for k in self.parameters.keys():
            if len(self.parameters[k]) != 1:
                raise ValueError(
                    f"Parameter '{k}' has invalid structure: expected only 'type' field"
                )
            if self.parameters[k]["type"] not in allowed_types:
                raise ValueError(
                    f"Parameter '{k}' has invalid type: '{self.parameters[k]['type']}'"
                )
        
        if self.returns["type"] not in allowed_types:
            raise ValueError("Invalide return type")
        
        return self

class Prompt(BaseModel):
    prompt: str


class Parser:
    @staticmethod
    def parse_func_file(path: str) -> list[FunctionDefinition]:
        functions: list[FunctionDefinition] = []
        try:
            with open(path) as file:
                data = json.load(file)
                functions = [FunctionDefinition(**func) for func in data]
                return functions

        except IOError as e:
            print(f"Error occured while opening file {path}: {e}")
            exit(1)

        except ValidationError as e:
            print(f"Validation error: {e}")
            exit(1)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in {path}: {e}")
            exit(1)

    @staticmethod
    def parse_input_file(path: str) -> list[Prompt]:
        prompts: list[Prompt] = []

        try:
            with open(path) as file:
                data =  json.load(file)
                prompts = [Prompt(**prompt) for prompt in data]
                print(data[2])
                return prompts
        
        except IOError as e:
            print(f"Error occured while opening file {path}: {e}")
            exit(1)

        except (ValidationError, KeyError) as e:
            print(f"Validation error: {e}")
            exit(1)
        
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in {path}: {e}")
            exit(1)