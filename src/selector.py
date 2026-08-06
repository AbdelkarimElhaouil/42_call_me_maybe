from .models import FunctionDefinition
from llm_sdk import Small_LLM_Model

class Selector:
    def __init__(self, prompts: list[str], functions: list[FunctionDefinition]):
        self.prompts = prompts
        self.functions = functions
        self.model = Small_LLM_Model()

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

    def __encode_available_func_names(self) -> list[list[int]]:
        func_names = [f.name for f in self.functions]
        func_names.append("none")
        func_name_tokenized = []

        for n in func_names:
            tokenized_name = self.model.encode(n).tolist()
            func_name_tokenized.append(tokenized_name)
        tokenized_name = self.model.encode("none")
        func_name_tokenized.append(tokenized_name)
        return func_name_tokenized

    def select_func_name(self):
        func_names_tokenized = self.__encode_available_func_names()
        i = 0
        for p in self.prompts:
            prompt = self.__construct_prompt(p)
            tokens = self.model.encode(prompt).tolist()
            allowed_tokens = []
            self.model.get_logits_from_input_ids(tokens)