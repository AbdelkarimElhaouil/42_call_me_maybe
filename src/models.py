"""Define Pydantic models used to validate project input data.

This module contains the models used to validate function definitions
and user prompts loaded from the project's JSON input files.
"""

from pydantic import (
    BaseModel,
    model_validator,
    ConfigDict,
    Field,
)


class FunctionDefinition(BaseModel):
    """Represent a function definition from functions_definition.json.

    A function definition contains the function name, its description,
    its parameters, and its return type. The model also validates the
    structure and types of the parameters and return value.

    Attributes:
        name: Name of the function.
        description: Description of what the function does.
        parameters: Mapping of parameter names to their type definitions.
        returns: Definition of the function's return type.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_min_length=1,
    )

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]

    @model_validator(mode="after")
    def validate_types(self) -> "FunctionDefinition":
        """Validate parameter and return type definitions.

        Each parameter must contain exactly one key named ``type``, and
        its value must be one of the supported types. The return definition
        must follow the same structure.

        The supported types are:

        - ``number``
        - ``integer``
        - ``object``
        - ``boolean``
        - ``string``

        Returns:
            The validated ``FunctionDefinition`` instance.

        Raises:
            ValueError: If a parameter or return definition has an invalid
                structure or contains an unsupported type.
            KeyError: If a parameter or return definition does not contain
                the required ``type`` key.
        """
        valid_types = [
            "number",
            "integer",
            "object",
            "boolean",
            "string",
        ]

        for k in self.parameters.keys():
            if len(self.parameters[k]) != 1:
                raise ValueError(
                    f"Parameter '{k}' has invalid structure: "
                    "expected only 'type' field"
                )

            try:
                self.parameters[k]["type"]
            except KeyError:
                raise KeyError(
                    f"The key of {k} parameter must be exactly 'type'"
                )

            if self.parameters[k]["type"].lower() not in valid_types:
                raise ValueError(
                    f"Type {self.parameters[k]['type']} is invalid\n"
                    f"Valid types are: {valid_types}"
                )

        if len(self.returns) != 1:
            raise ValueError(
                "The returns should contain only one type"
            )

        try:
            self.returns["type"]
        except KeyError:
            raise KeyError(
                "The key in returns must be exactly 'type'"
            )

        if self.returns["type"].lower() not in valid_types:
            raise ValueError(
                f"Type {self.returns['type']} is invalid\n"
                f"Valid types are: {valid_types}"
            )

        return self


class Prompt(BaseModel):
    """Represent a user prompt loaded from an input file.

    The model validates that the input object contains only the expected
    fields and stores the user's prompt.

    Attributes:
        prompt: Text of the user's request.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str