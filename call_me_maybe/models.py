from pydantic import BaseModel, ValidationError, model_validator, ConfigDict, Field

class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_config = ConfigDict(str_strip_whitespace=True)
    name: str = Field(min_length=1)
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
    model_config = ConfigDict(extra="forbid")
    prompt: str