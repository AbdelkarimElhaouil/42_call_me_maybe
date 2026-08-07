from .models import FunctionDefinition
from numpy import argmax, inf
from llm_sdk import Small_LLM_Model
from .utils import load_vocabe

class Selector:
    def __init__(self, prompts: list[str], functions: list[FunctionDefinition]):
        self.prompts = prompts
        self.functions = functions
        self.model = Small_LLM_Model()
        # self.vocab = load_vocabe(self.model.get_path_to_vocab_file())

    def __construct_prompt(self, user_prompt: str) -> str:
        prompt = (
            "You are a function selector. Given the user request, "
            "choose the most appropriate function from the list below. "
            "Output only the function name, nothing else.\n"
            "Available functions:\n"
        )
        for f in self.functions:
            prompt += f"- {f.name}: {f.description}\n"
        prompt += "- none: No available function matches the user request.\n\n"
        prompt += f"User request: {user_prompt}\n\n"
        prompt += "Function to call:\n"

        return prompt

    def test(self):
        # Encode in isolation
        isolated = self.model.encode("fn_add_numbers").tolist()

        # Encode in context (after your prompt ending)
        full = self.model.encode('Function to call:\nfn_add_numbers').tolist()

        # Compare: does the function name portion match?
        print(isolated)
        print(full[len(isolated):])

    # def __encode_available_func_names(self) -> list[list[int]]:
    #     func_names = [f.name for f in self.functions]
    #     func_names.append("none")
    #     func_name_tokenized = []

    #     for n in func_names:
    #         tokenized_name = self.model.encode(n).tolist()
    #         func_name_tokenized.append(tokenized_name)
    #     tokenized_name = self.model.encode("none")
    #     func_name_tokenized.append(tokenized_name)
    #     return func_name_tokenized

    # def __get_allowed_tokens(
    #         self, generated_ids: list[int], tokens: list[list[int]], state: int
    #     ) -> list[int]:
    #     if not generated_ids:
    #         return [t[0] for t in tokens]
    #     else :
    #         gen_ids_len = len(generated_ids)
    #         return [t[state] for t in tokens if t[:gen_ids_len] == generated_ids]

    # def __choose_max_token(self, logits: list[int], allowed_tokens_ids: list[int]) -> int:
    #     candidates = {str(id):logits[id] for id in allowed_tokens_ids}
    #     max_token = -inf
    #     for k in candidates.keys():
    #         if candidates[k] > max_token:
    #             id = k
    #             max_token = candidates[k]

    #     return int(id)


    # def __get_generated_func_name(
    #         func_names_tokenized: list[list[int]], generated_ids: list[int]
    #         ) -> list[int]:
    #     generated_func_name = [
    #         f for f in func_names_tokenized if f == generated_ids
    #     ]
    #     return generated_func_name[0]

    # def select_func_name(self):
    #     func_names_tokenized = self.__encode_available_func_names()
    #     for p in self.prompts:
    #         prompt = self.__construct_prompt(p)
    #         tokens = self.model.encode(prompt).tolist()
    #         generated_ids = []
    #         state = 0
    #         while True:
    #             allowed_tokens = self.__get_allowed_tokens(
    #                 func_names_tokenized, generated_ids, state
    #             )
    #             logits = self.model.get_logits_from_input_ids(tokens)
    #             chosen_token = self.__choose_max_token(logits, allowed_tokens)
    #             generated_ids.append(chosen_token)
    #             generated_func_name = self.__get_generated_func_name(func_names_tokenized, generated_ids)
    #             if generated_func_name:
    #                 break
    #             state += 1