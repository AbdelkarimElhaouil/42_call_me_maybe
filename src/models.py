from pydantic import BaseModel, model_validator, ConfigDict, Field, version

class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", str_min_length=1)
    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]

    @model_validator(mode="after")
    def validate_types(self) -> "FunctionDefinition":
        valid_types  = ["number", "integer", "object", "boolean", "string"]
        for k in self.parameters.keys():
            if len(self.parameters[k]) != 1:
                raise ValueError(
                    f"Parameter '{k}' has invalid structure: expected only 'type' field"
                )
            try :
                self.parameters[k]["type"]
            except KeyError:
                raise KeyError(
                    f"The key of {k} parameter must be exactly 'type'"
                )
            if self.parameters[k]["type"].lower() not in valid_types:
                raise ValueError(f"Type {self.parameters[k]["type"]} is invalid\n"
                                 f"Valide types are: {valid_types}")
        if len(self.returns) != 1:
            raise ValueError(
                "The returns should conatain only one type"
            )
        try :
            self.returns["type"]
        except:
            raise KeyError(
                f"The key in returns must be exactly 'type'"
            )
        if self.returns["type"].lower() not in valid_types:
            raise ValueError(f"Type {self.parameters[k]["type"]} is invalid\n"
                             f"Valide types are: {valid_types}")
        return self

class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str