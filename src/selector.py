from .models import FunctionDefinition
from numpy import argmax, inf
from llm_sdk import Small_LLM_Model
from sys import exit

class Selector:
    def __init__(self, prompts: list[str], functions: list[FunctionDefinition]):
        self.prompts = prompts
        self.functions = functions
        self.model = Small_LLM_Model()
        self.__allowed_tokens: dict[str, dict[str, list[int]] | list[int]] = {
            "integer": {
                "allow": [
                    self.model.encode(c).tolist()[0][0] for c in "0123456789"
                ],
                "stop": [
                    self.model.encode(c).tolist()[0][0] for c in "},.\n"
                ]
            },
            "number":{
                "allow": [
                    self.model.encode(c).tolist()[0][0] for c in "0123456789."
                ],
                "stop": [
                    self.model.encode(c).tolist()[0][0] for c in "},\n"
                ]
            },
            "boolean": {
                "allow": [
                    self.model.encode(c).tolist()[0][0] for c in ["true", "false"]
                ]
            },
        }
        self.func_names_tokenized = self.__encode_available_func_names()

    def __construct_prompt(self, user_prompt: str) -> str:
        prompt = "Task: Map the user request to the correct function name.\n\n"
        for f in self.functions:
            prompt += f"- {f.name}: {f.description}\n"
        prompt += "- fn_no_function: select it when the user request dont match the functions above.\n"
        prompt += f"User request: {user_prompt}\n\n"
        prompt += "Function to map: \n"

        return prompt

    def __encode_available_func_names(self) -> list[list[int]]:
        func_names = [f.name for f in self.functions]
        func_names.append("fn_no_function")
        func_name_tokenized = []

        for n in func_names:
            tokenized_name = self.model.encode(n).tolist()[0]
            func_name_tokenized.append(tokenized_name)
        return func_name_tokenized

    def __get_allowed_func_tokens(
            self, generated_ids: list[int], tokens: list[list[int]]
        ) -> list[int]:
        if not generated_ids:
            return [t[0] for t in tokens]
        else :
            gen_ids_len = len(generated_ids)
            return [t[gen_ids_len] for t in tokens if t[:gen_ids_len] == generated_ids]


    def __choose_max_token(self, logits: list[float], allowed_tokens_ids: list[int]) -> int:
        candidates = {str(id):logits[id] for id in allowed_tokens_ids}
        max_token = -inf
        for k in candidates.keys():
            if candidates[k] > max_token:
                id = k
                max_token = candidates[k]
        return int(id)


    def __select_func_name(self, user_prompt: str) -> str:
        prompt = self.__construct_prompt(user_prompt)
        input_ids: list[int] = self.model.encode(prompt).tolist()[0]
        generated_ids = []
        while True:
            allowed_tokens = self.__get_allowed_func_tokens(
                generated_ids, self.func_names_tokenized
            )
            logits = self.model.get_logits_from_input_ids(input_ids)
            chosen_token = self.__choose_max_token(logits, allowed_tokens)
            generated_ids.append(chosen_token)
            input_ids.append(chosen_token)
            if generated_ids in self.func_names_tokenized:
                res = self.model.decode(generated_ids)
                return res

    def __extracte_func_params(self, func_name: str) -> dict[str, str]:
        params = {}
        for f in self.functions:
            if f.name == func_name:
                for param_name in f.parameters:
                    params.update({param_name:f.parameters[param_name]["type"]})
                return params


    def generate_answers(self) -> list[dict[str, str]]:
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
                    "parameters": params_values
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
            self, user_prompt: str, params: dict[str, str]
        ) -> dict[str, str]:
        prompt = f"Task: extract the values of parameters from the user request.\n"
        prompt += f"user request: {user_prompt}\n"
        prompt_ids: list[int] = self.model.encode(prompt).tolist()[0]
        result = {}

        for p in params.keys():
            param_type = params[p]
            line_1 = f"\nParameter name: {p}"
            line_2 = f"\nParameter value: "
            prompt_ids.extend(self.model.encode(line_1 + line_2).tolist()[0])

            if param_type == "integer":
                val = self.__extract_int_value(prompt_ids)
                result.update({p:val})

            elif param_type == "number":
                val = self.__extract_float_value(prompt_ids)
                result.update({p:val})

            elif param_type == "boolean":
                val = self.__extract_bool_value(prompt_ids)
                result.update({p: val})

            else :
                val = self.__extract_str_value(prompt_ids)
                result.update({p: val})

        return result
    
    def __extract_str_value(self, prompt_ids: list[int]) -> str:
        prefix = self.model.encode("\"").tolist()[0]
        prompt_ids.extend(prefix)
        gen_ids = []
        while True:
            logits = self.model.get_logits_from_input_ids(prompt_ids)
            chosen_token = argmax(logits)
            token_decoded = self.model.decode(chosen_token)
            prompt_ids.append(chosen_token)
            if "\"" in token_decoded:
                val = self.model.decode(gen_ids)
                val += token_decoded.split(sep="\"")[0]
                return val.strip()
            gen_ids.append(chosen_token)

    def __extract_bool_value(self, prompt_ids: list[int]) -> bool:
        allow_tokens = self.__allowed_tokens["boolean"]["allow"]
        logits = self.model.get_logits_from_input_ids(prompt_ids)
        chosen_token = self.__choose_max_token(logits, allow_tokens)
        prompt_ids.append(chosen_token)
        return self.model.decode(chosen_token)
    
    def __extract_float_value(self, prompt_ids: list[int]) -> int:
        allow_tokens = self.__allowed_tokens["number"]["allow"]
        stop_tokens = self.__allowed_tokens["number"]["stop"]
        sign_tokens = [
            self.model.encode(c).tolist()[0][0] for c in [" ", " -"]
        ]
        state = 1
        point_appearance = 0
        gen_ids = []
        while True:
            if  state:
                allowed = sign_tokens
                state = 0
            else:
                allowed = allow_tokens + stop_tokens
            logits = self.model.get_logits_from_input_ids(prompt_ids)
            chosen_token = self.__choose_max_token(logits, allowed)
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
        allow_tokens = self.__allowed_tokens["integer"]["allow"]
        stop_tokens = self.__allowed_tokens["integer"]["stop"]
        sign_tokens = [
            self.model.encode(c).tolist()[0][0] for c in [" ", " -"]
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
            chosen_token = self.__choose_max_token(logits, allowed)
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
