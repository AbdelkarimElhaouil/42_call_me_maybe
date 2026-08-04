from models import FunctionDefinition
from llm_sdk import Small_LLM_Model as llm_model
class Generator:
    def __init__(self, prompts: list[str], functions: list[FunctionDefinition]):
        self.prompts = prompts
        self.functions = functions
        self.model = llm_model
        
    def generate(self):
