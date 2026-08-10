from json import load, dump
from pathlib import Path

def load_vocabe(path: str) -> dict[str, str]:
    try:
        with open(path) as f:
            vocab_dict = load(f)
            return vocab_dict
    except IOError as e:
        print(f"Error while opening a file: {e}")
        exit(1)


def dump_result(output_path: str, result: list[dict[str, str]]) -> None:
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as output_file:
            dump(result, output_file, indent=4)
    except IOError as e:
        print(f"Error while opening a file: {e}")