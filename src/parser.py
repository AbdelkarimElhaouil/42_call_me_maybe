"""Provide parsing utilities for the project's input files.

This module reads JSON files, validates their contents using Pydantic
models, and converts the data into structured Python objects.

It also handles file, JSON, and validation errors by displaying an
appropriate error message and terminating the program with a non-zero
exit code.
"""

from sys import exit
from pydantic import ValidationError
from .models import FunctionDefinition, Prompt
from json import load, JSONDecodeError


class Parser:
    """Group file-parsing utilities used by the project.

    The class provides static methods for parsing command-line arguments,
    function-definition files, and prompt files.
    """

    @staticmethod
    def parse_cli_args(argv: list[str]) -> dict[str, str]:
        """Parse command-line arguments and return their values.

        The method supports arguments for specifying the input file,
        output file, and function-definition file. Default paths are used
        when the corresponding arguments are not provided.

        Args:
            argv: Command-line arguments passed to the program.

        Returns:
            A dictionary containing the paths associated with the supported
            command-line arguments.

        Raises:
            SystemExit: If an unsupported argument is provided or a required
                argument value is missing.
        """
        argv_len = len(argv)

        valid_args = [
            "--input",
            "--output",
            "--functions_definition",
        ]

        default_result = {
            "--input": "data/input/function_calling_tests.json",
            "--output": "data/output/function_calling_results.json",
            "--functions_definition": (
                "data/input/functions_definition.json"
            ),
        }

        if argv_len > 1:
            i = 1

            while i < argv_len:
                if argv[i] in valid_args and argv_len > i + 1:
                    default_result.update(
                        {argv[i]: argv[i + 1]}
                    )
                    i += 2

                else:
                    print(
                        "Error: please provide valid command line arguments"
                    )
                    print("Usage: uv run python -m src", end="")
                    print(
                        "[--functions_definition "
                        "<function_definition_file>]",
                        end="",
                    )
                    print(
                        "[--input <input_file>] "
                        "[--output <output_file>]"
                    )
                    exit(1)

        return default_result

    @staticmethod
    def parse_func_file(
        path: str,
    ) -> list[FunctionDefinition]:
        """Parse a JSON file containing function definitions.

        Each function definition is validated using the
        ``FunctionDefinition`` Pydantic model.

        Args:
            path: Path to the JSON file containing function definitions.

        Returns:
            A list of validated ``FunctionDefinition`` objects.

        Raises:
            SystemExit: If the file cannot be opened, contains invalid JSON,
                or contains data that fails validation.
        """
        functions: list[FunctionDefinition] = []

        try:
            with open(path) as file:
                data = load(file)
                functions = [
                    FunctionDefinition(**func)
                    for func in data
                ]
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
        """Parse a JSON file containing user prompts.

        Each entry is validated using the ``Prompt`` Pydantic model.
        Empty or whitespace-only prompts are removed from the result.

        Args:
            path: Path to the JSON file containing prompts.

        Returns:
            A list containing the non-empty prompt strings.

        Raises:
            SystemExit: If the file cannot be opened, contains invalid JSON,
                contains invalid prompt data, or contains no valid prompts.
        """
        prompts: list[Prompt] = []

        try:
            with open(path) as file:
                data = load(file)
                prompts = [
                    Prompt(**prompt)
                    for prompt in data
                ]

                valid_prompts = [
                    p.prompt
                    for p in prompts
                    if p.prompt.strip() != ""
                ]

                if not valid_prompts:
                    print(
                        "All prompts are empty: "
                        "please provide a valid prompt."
                    )
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
