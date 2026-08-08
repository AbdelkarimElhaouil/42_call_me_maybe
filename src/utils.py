from json import load
from sys import exit

def load_vocabe(path: str) -> dict[str, str]:
    try:
        with open(path) as f:
            vocab_dict = load(f)
            return vocab_dict
    except IOError as e:
        print(f"Error while opening a file: {e}")
        exit(1)