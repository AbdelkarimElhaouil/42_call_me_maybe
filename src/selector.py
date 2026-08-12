"""Function selection and parameter extraction using a small language model.

This module provides the ``Selector`` class, which maps user prompts to
available function definitions and extracts the corresponding parameter
values from the prompts.

The selector uses constrained token generation to restrict the language
model's output to valid function names and parameter types.
"""

from .models import FunctionDefinition
from numpy import argmax, inf
from llm_sdk import Small_LLM_Model
from sys import exit


class Selector:
    """Select functions and extract their parameters from user prompts.

    The selector uses a small language model to perform two main tasks:

    1. Map each user prompt to one of the available functions.
    2. Extract the values of the selected function's parameters.

    Token generation is constrained according to the expected output type
    of each parameter. For example, integer parameters are restricted to
    numeric tokens, while boolean parameters are restricted to ``true`` and
    ``false``.

    Args:
        prompts: User requests that should be mapped to functions.
        functions: Available function definitions that the model can select.
    """

    def __init__(
        self,
        prompts: list[str],
        functions: list[FunctionDefinition],
    ):
        """Initialize the selector.

        Args:
            prompts: User requests to process.
            functions: Functions available for selection.
        """
        self.prompts = prompts
        self.functions = functions
        self.model = Small_LLM_Model()

        self.__allowed_tokens: dict[
            str, dict[str, list[int]] | list[int]
        ] = {
            "integer": {
                "allow": [
                    self.model.encode(c).tolist()[0][0]
                    for c in "0123456789"
                ],
                "stop": [
                    self.model.encode(c).tolist()[0][0]
                    for c in "},.\n"
                ],
            },
            "number": {
                "allow": [
                    self.model.encode(c).tolist()[0][0]
                    for c in "0123456789."
                ],
                "stop": [
                    self.model.encode(c).tolist()[0][0]
                    for c in "},\n"
                ],
            },
            "boolean": {
                "allow": [
                    self.model.encode(c).tolist()[0][0]
                    for c in ["true", "false"]
                ]
            },
        }

        self.func_names_tokenized = self.__encode_available_func_names()

    def __construct_prompt(self, user_prompt: str) -> str:
        """Construct the prompt used to select a function.

        The generated prompt contains the available function names and
        descriptions, followed by the user's request. A special
        ``fn_no_function`` option is added for requests that do not match
        any available function.

        Args:
            user_prompt: User request that needs to be mapped.

        Returns:
            The formatted prompt sent to the language model.
        """
        prompt = (
            "Task: Map the user request to the correct function name.\n\n"
        )

        for f in self.functions:
            prompt += f"- {f.name}: {f.description}\n"

        prompt += (
            "- fn_no_function: select it when the user request "
            "dont match the functions above.\n"
        )
        prompt += f"User request: {user_prompt}\n\n"
        prompt += "Function to map: \n"

        return prompt

    def __encode_available_func_names(self) -> list[list[int]]:
        """Encode all available function names into model tokens.

        ``fn_no_function`` is also encoded and included as a valid selection.

        Returns:
            A list containing the token IDs for each available function name.
        """
        func_names = [f.name for f in self.functions]
        func_names.append("fn_no_function")

        func_name_tokenized = []

        for name in func_names:
            tokenized_name = self.model.encode(name).tolist()[0]
            func_name_tokenized.append(tokenized_name)

        return func_name_tokenized

    def __get_allowed_func_tokens(
        self,
        generated_ids: list[int],
        tokens: list[list[int]],
    ) -> list[int]:
        """Get tokens that can legally follow the generated function name.

        The method compares the tokens already generated against the
        beginning of every available function name and returns the possible
        next tokens.

        Args:
            generated_ids: Tokens already generated for the function name.
            tokens: Tokenized available function names.

        Returns:
            Token IDs that are valid at the current generation position.
        """
        if not generated_ids:
            return [t[0] for t in tokens]

        gen_ids_len = len(generated_ids)

        return [
            t[gen_ids_len]
            for t in tokens
            if t[:gen_ids_len] == generated_ids
        ]

    def __choose_max_token(
        self,
        logits: list[float],
        allowed_tokens_ids: list[int],
    ) -> int:
        """Choose the highest-probability token among allowed tokens.

        Args:
            logits: Model logits for the next token.
            allowed_tokens_ids: Token IDs that are allowed to be selected.

        Returns:
            The ID of the allowed token with the highest logit.
        """
        candidates = {
            str(token_id): logits[token_id]
            for token_id in allowed_tokens_ids
        }

        max_token = -inf

        for key in candidates.keys():
            if candidates[key] > max_token:
                token_id = key
                max_token = candidates[key]

        return int(token_id)

    def __select_func_name(self, user_prompt: str) -> str:
        """Select the function that best matches a user request.

        Function-name generation is constrained so that the model can only
        generate a prefix of one of the available function names.

        Args:
            user_prompt: User request to classify.

        Returns:
            The name of the selected function.
        """
        prompt = self.__construct_prompt(user_prompt)
        input_ids: list[int] = self.model.encode(prompt).tolist()[0]
        generated_ids = []

        while True:
            allowed_tokens = self.__get_allowed_func_tokens(
                generated_ids,
                self.func_names_tokenized,
            )

            logits = self.model.get_logits_from_input_ids(input_ids)

            chosen_token = self.__choose_max_token(
                logits,
                allowed_tokens,
            )

            generated_ids.append(chosen_token)
            input_ids.append(chosen_token)

            if generated_ids in self.func_names_tokenized:
                return self.model.decode(generated_ids)

    def __extracte_func_params(
        self,
        func_name: str,
    ) -> dict[str, str]:
        """Extract parameter names and types for a function.

        Args:
            func_name: Name of the selected function.

        Returns:
            A dictionary mapping parameter names to their declared types.
        """
        params = {}

        for f in self.functions:
            if f.name == func_name:
                for param_name in f.parameters:
                    params.update(
                        {
                            param_name: f.parameters[param_name]["type"]
                        }
                    )

                return params

    def generate_answers(self) -> list[dict[str, str]]:
        """Process all prompts and generate function-selection results.

        Each result contains the original prompt, the selected function
        name, and the extracted parameter values.

        Returns:
            A list of dictionaries containing the results for every prompt.
        """
        result = []
        i = 1

        for p in self.prompts:
            func_name = self.__select_func_name(p)

            if func_name == "fn_no_function":
                params_values = {}
            else:
                params = self.__extracte_func_params(func_name)
                params_values = self.__select_params_values(p, params)

            result.append(
                {
                    "prompt": p,
                    "name": func_name,
                    "parameters": params_values,
                }
            )

            print(f"Result {i}:")
            print("User prompt:", p)
            print("Function name:", func_name)
            print("Parameters:", params_values)
            print("\n")

            i += 1

        return result

    def __select_params_values(
        self,
        user_prompt: str,
        params: dict[str, str],
    ) -> dict[str, str]:
        """Extract parameter values from a user request.

        Each parameter is processed according to its declared type.
        Integer, floating-point, boolean, and string values use different
        token-generation strategies.

        Args:
            user_prompt: User request containing the parameter values.
            params: Mapping of parameter names to their expected types.

        Returns:
            A dictionary mapping parameter names to extracted values.
        """
        prompt = (
            "Task: extract the values of parameters "
            "from the user request.\n"
        )
        prompt += f"user request: {user_prompt}\n"

        prompt_ids: list[int] = self.model.encode(prompt).tolist()[0]
        result = {}

        for p in params.keys():
            param_type = params[p]

            line_1 = f"\nParameter name: {p}"
            line_2 = "\nParameter value: "

            prompt_ids.extend(
                self.model.encode(line_1 + line_2).tolist()[0]
            )

            if param_type == "integer":
                val = self.__extract_int_value(prompt_ids)
                result.update({p: val})

            elif param_type == "number":
                val = self.__extract_float_value(prompt_ids)
                result.update({p: val})

            elif param_type == "boolean":
                val = self.__extract_bool_value(prompt_ids)
                result.update({p: val})

            else:
                val = self.__extract_str_value(prompt_ids)
                result.update({p: val})

        return result

    def __extract_str_value(self, prompt_ids: list[int]) -> str:
        """Extract a string value from the model output.

        String generation starts with a quotation mark and continues until
        another quotation mark is generated.

        Args:
            prompt_ids: Tokenized prompt used as input to the model.

        Returns:
            The extracted string value.
        """
        prefix = self.model.encode('"').tolist()[0]
        prompt_ids.extend(prefix)

        gen_ids = []

        while True:
            logits = self.model.get_logits_from_input_ids(prompt_ids)
            chosen_token = argmax(logits)
            token_decoded = self.model.decode(chosen_token)

            prompt_ids.append(chosen_token)

            if '"' in token_decoded:
                val = self.model.decode(gen_ids)
                val += token_decoded.split(sep='"')[0]
                return val.strip()

            gen_ids.append(chosen_token)

    def __extract_bool_value(self, prompt_ids: list[int]) -> bool:
        """Extract a boolean value from the model output.

        Only tokens representing ``true`` or ``false`` are allowed.

        Args:
            prompt_ids: Tokenized prompt used as input to the model.

        Returns:
            The extracted boolean value.
        """
        allow_tokens = self.__allowed_tokens["boolean"]["allow"]

        logits = self.model.get_logits_from_input_ids(prompt_ids)

        chosen_token = self.__choose_max_token(
            logits,
            allow_tokens,
        )

        prompt_ids.append(chosen_token)

        return self.model.decode(chosen_token)

    def __extract_float_value(self, prompt_ids: list[int]) -> int:
        """Extract a floating-point number from the model output.

        The generated value may contain digits, a decimal point, and a
        negative sign. Generation stops when a stop token is encountered or
        when more than one decimal point is produced.

        Args:
            prompt_ids: Tokenized prompt used as input to the model.

        Returns:
            The extracted floating-point value.
        """
        allow_tokens = self.__allowed_tokens["number"]["allow"]
        stop_tokens = self.__allowed_tokens["number"]["stop"]

        sign_tokens = [
            self.model.encode(c).tolist()[0][0]
            for c in [" ", " -"]
        ]

        state = 1
        point_appearance = 0
        gen_ids = []

        while True:
            if state:
                allowed = sign_tokens
                state = 0
            else:
                allowed = allow_tokens + stop_tokens

            logits = self.model.get_logits_from_input_ids(prompt_ids)

            chosen_token = self.__choose_max_token(
                logits,
                allowed,
            )

            token_decoded = self.model.decode(chosen_token)
            prompt_ids.append(chosen_token)

            if token_decoded == ".":
                point_appearance += 1

            if point_appearance == 2 or chosen_token in stop_tokens:
                break

            gen_ids.append(chosen_token)

        try:
            return float(self.model.decode(gen_ids))
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)

    def __extract_int_value(self, prompt_ids: list[int]) -> int:
        """Extract an integer value from the model output.

        Only digits and a possible negative sign are allowed during
        generation. Generation stops when an integer stop token is produced.

        Args:
            prompt_ids: Tokenized prompt used as input to the model.

        Returns:
            The extracted integer value.
        """
        allow_tokens = self.__allowed_tokens["integer"]["allow"]
        stop_tokens = self.__allowed_tokens["integer"]["stop"]

        sign_tokens = [
            self.model.encode(c).tolist()[0][0]
            for c in [" ", " -"]
        ]

        state = 1
        gen_ids = []

        while True:
            if state:
                allowed = sign_tokens
                state = 0
            else:
                allowed = allow_tokens + stop_tokens

            logits = self.model.get_logits_from_input_ids(prompt_ids)

            chosen_token = self.__choose_max_token(
                logits,
                allowed,
            )

            prompt_ids.append(chosen_token)

            if chosen_token in stop_tokens:
                break

            gen_ids.append(chosen_token)

        try:
            val = int(self.model.decode(gen_ids))
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)

        return val