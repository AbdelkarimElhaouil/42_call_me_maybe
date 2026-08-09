from .models import FunctionDefinition
from numpy import argmax, inf
from llm_sdk import Small_LLM_Model
from .utils import load_vocabe
from sys import exit

class Selector:
    def __init__(self, prompts: list[str], functions: list[FunctionDefinition]):
        self.prompts = prompts
        self.functions = functions
        self.model = Small_LLM_Model()
        self.vocab = load_vocabe(self.model.get_path_to_vocab_file())
        self.__allowed_tokens: dict[str, dict[str, list[int]] | list[int]] = {
            "integer": {
                "allow": self.__get_allowed_type_tokens("integer"),
                "stop": [
                    self.model.encode(c).tolist()[0][0] for c in "},.\n"
                ]
            },
            "number":{
                "allow": self.__get_allowed_type_tokens("number"),
                "stop": [
                    self.model.encode(c).tolist()[0][0] for c in "},\n"
                ]
            },
            "boolean": {
                "allow": self.__get_allowed_type_tokens("boolean")
            },
        }
        self.func_names_tokenized = self.__encode_available_func_names() # lust[list[int]]
        # self.allowed_number_tokens = self.__get_allowed_type_tokens("number")
        # self.allowed_boolean_tokens = self.__get_allowed_type_tokens("boolean")
        # self.allowed_int_tokens = self.__get_allowed_type_tokens("integer")

    def __get_allowed_type_tokens(self, type: str) -> list[int]:
        valid_set: set = {}
        if type.lower() == "boolean":
            return [self.model.encode(t).tolist()[0][0] for t in ["true", "false"]] # [[1]] = [[1]]
        if type.lower() == "number":
            valid_set = set("0123456789.")
        elif type.lower() == "integer":
            valid_set = set("0123456789")
        allowed_tokens = []
        for c in valid_set:
            token = self.model.encode(c).tolist()[0][0]
            allowed_tokens.append(token)
        
        return allowed_tokens


    def __construct_prompt(self, user_prompt: str) -> str:
        # prompt = (
        #     "You are a function selector. Given the user request, "
        #     "choose the most appropriate function from the list below. "
        #     "Output only the function name, nothing else.\n"
        #     "Available functions:\n"
        # )
        prompt = "Task: Map the user request to the correct function name.\n\n"
        for f in self.functions:
            prompt += f"- {f.name}: {f.description}\n"
        prompt += "- fn_null_function: choose it when no function matches the request.\n\n"
        prompt += f"User request: {user_prompt}\n\n"
        prompt += "Function to map:\n"

        return prompt

    def __encode_available_func_names(self) -> list[list[int]]:
        func_names = [f.name for f in self.functions]
        func_names.append("fn_null_function")
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
            print(self.model.decode(int(k)), "=", candidates[k])
            if candidates[k] > max_token:
                id = k
                max_token = candidates[k]
        print("-"*100)
        return int(id)


    def __get_generated_func_name(
            self, func_names_tokenized: list[list[int]], generated_ids: list[int]
            ) -> list[int] | None:
        for f in func_names_tokenized:
            if f == generated_ids:
                return f
        return None


    def __construct_param_prompt(self, user_prompt: str) -> str:
        prompt = f"Extract the value of parameter from the user request.\n"
        prompt += f"User request: \"{user_prompt}\"\n"
        return prompt


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
            generated_func_name = self.__get_generated_func_name(self.func_names_tokenized, generated_ids)
            if generated_func_name:
                res = self.model.decode(generated_ids)
                print("RESULT:", "="*100, "\n", user_prompt, "->", res, "\n", "="*100)
                return res
    

    def generate_json(self):
        for p in self.prompts:
            func = self.__select_func_name(p)
            params = self.__extracte_func_params(func)
            print(params)
            res = self.__select_params_values(p, params)
            d = {
                "prompt": p,
                "name": func,
                "parameters": res
            }
            print(d)


    def __select_params_values(
            self, user_prompt: str, params: dict[str, str]
        ) -> dict[str, str]:
        prompt = self.__construct_param_prompt(user_prompt)
        prompt_ids: list[int] = self.model.encode(prompt).tolist()[0]
        result = {}
    
        for p in params.keys():
            line_1 = f"\nParameter name: {p}"
            line_2 = f"\nParameter value: "
            prompt_ids.extend(self.model.encode(line_1 + line_2).tolist()[0])
            param_type = params[p]
    
            if param_type == "integer":
                val = self.__extract_int_value(prompt_ids)
                result.update({p:val})
                print("*"*80, self.model.decode(prompt_ids), "\n", "*"*80)

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
                return self.model.decode(gen_ids)
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

            if token_decoded == ".":
                point_appearance += 1
            
            if point_appearance == 2:
                prompt_ids.append(chosen_token)
                break

            print("*"*80, self.model.decode(prompt_ids), "\n")
            prompt_ids.append(chosen_token)
            if chosen_token in stop_tokens:
                break
            gen_ids.append(chosen_token)

        try:
            val = float(self.model.decode(gen_ids))
        except ValueError as e:
                print(f"Error: {e}")
                exit(1)
        return val


    def __extract_int_value(self, prompt_ids: list[int]) -> int:
        allow_tokens = self.__allowed_tokens["integer"]["allow"]
        stop_tokens = self.__allowed_tokens["integer"]["stop"]
        sign_tokens = [
            self.model.encode(c).tolist()[0][0] for c in [" ", " -"]
        ]
        state = 1
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
            print("*"*80, self.model.decode(prompt_ids), "\n")
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

    def __extracte_func_params(self, func_name: str) -> dict[str, str]:
        params = {}
        for f in self.functions:
            if f.name == func_name:
                for param_name in f.parameters:
                    params.update({param_name:f.parameters[param_name]["type"]})
                return params


    def select_params(self):
        for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
            print(self.model.enco).tolist()[0][0]
        print(self.model.encode("true").tolist()[0][0])
        print(self.model.encode("false"))