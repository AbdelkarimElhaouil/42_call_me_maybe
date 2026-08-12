from json import dump
from pathlib import Path


def dump_result(
        output_path: str,
        result: list[dict[str, str | dict[str, str | int | float]]]
        ) -> None:
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as output_file:
            dump(result, output_file, indent=4)
    except IOError as e:
        print(f"Error while opening a file: {e}")
