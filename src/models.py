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
        for k in self.parameters.keys():
            if len(self.parameters[k]) != 1:
                raise ValueError(
                    f"Parameter '{k}' has invalid structure: expected only 'type' field"
                )
            try :
                self.parameters[k]["type"]
            except:
                raise ValueError(
                    f"The key in parameters must be exactly 'type'"
                )
        if len(self.returns) != 1:
            raise ValueError(
                "The returns should conatain only one type"
            )
        try :
            self.returns["type"]
        except:
            raise ValueError(
                f"The key in returns must be exactly 'type'"
            )
        return self

class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str