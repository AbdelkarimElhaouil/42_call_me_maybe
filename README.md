*This project has been created as part of the 42 curriculum by aelhaoui.*

## Description

This project implements a **function-calling system powered by a small language model (LLM)**.

The goal is to take natural-language user requests, determine which predefined function best matches the request, and extract the values of the function's parameters.

For example, given a request such as:

```text
Set the temperature to 25 degrees.
```

and a set of available function definitions, the system can determine the appropriate function and extract:

```text
Function: set_temperature
Parameter: temperature = 25
```

The project is based on **constrained decoding**. Instead of allowing the language model to generate arbitrary text, the implementation restricts the tokens that can be selected at each generation step. This makes the model's output conform to the expected function names and parameter types.

The main components are:

* **Parser** — Parses command-line arguments and JSON input files.
* **Pydantic models** — Validate function definitions and prompts.
* **Selector** — Uses the language model to select functions and extract parameters.
* **Constrained decoding** — Restricts model output to valid tokens according to the expected result.
* **Result utilities** — Store the generated function-calling results in an output file.

---


### Requirements

The project requires:

* Python 3
* `uv`
* The project's LLM SDK
* Pydantic
* NumPy

The exact Python version and dependencies should be taken from the project's `pyproject.toml`.

### Installation

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd <project-directory>
```

Install the project dependencies using `uv`:

```bash
uv sync
```

### Input files

By default, the program expects the following files:

```text
data/
├── input/
│   ├── functions_definition.json
│   └── function_calling_tests.json
└── output/
    └── function_calling_results.json
```

The function definition file describes the functions that the model is allowed to select.

A simplified example:

```json
[
    {
        "name": "set_temperature",
        "description": "Set the temperature to a specified value.",
        "parameters": {
            "temperature": {
                "type": "number"
            }
        },
        "returns": {
            "type": "boolean"
        }
    }
]
```

The prompt file contains the user requests that should be processed.
## Instructions

### Execution

Run the project with:

```bash
uv run python3 -m src
```

The program also accepts custom input, output, and function-definition files:

```bash
uv run python3 -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```


## Algorithm Explanation

The core of the project is a **constrained decoding algorithm**.

A normal language model predicts a probability distribution over its vocabulary and can potentially generate any token. This project restricts that behavior so that only tokens that can lead to a valid result are considered.

### 1. Function selection

First, the program constructs a prompt containing the available functions:

```text
Task: Map the user request to the correct function name.

- function_a: Description of function A
- function_b: Description of function B
- fn_no_function: select it when the user request doesn't match...

User request: ...
Function to map:
```

The available function names are tokenized beforehand.

For example, if the available functions are:

```text
get_weather
send_email
calculate_sum
```

their tokenized representations are stored.

The model then generates the function name **one token at a time**.

At every step, the algorithm:

1. Looks at the tokens generated so far.
2. Finds function names whose token sequences match that prefix.
3. Extracts the possible next tokens from those function names.
4. Retrieves the model's logits.
5. Selects the highest-logit token among the allowed tokens.
6. Adds the selected token to the generated sequence.
7. Stops when the generated sequence exactly matches one of the available function names.

Conceptually:

```text
Available functions:

get_weather
get_time
send_email

Generated:
"get_"

Possible next tokens:
"weather" or "time"

The model chooses the highest-scoring valid continuation.
```

This prevents the model from producing a function name that does not exist.

The special:

```text
fn_no_function
```

option is included so that the model can explicitly indicate that no available function matches the request.

### 2. Parameter extraction

Once a function has been selected, the program retrieves its parameter definitions.

For example:

```json
{
    "temperature": {
        "type": "number"
    },
    "unit": {
        "type": "string"
    }
}
```

The parameter type determines the decoding strategy.

### Integer

For an integer parameter, the allowed tokens are restricted to:

```text
0 1 2 3 4 5 6 7 8 9
```

along with the appropriate sign and stop tokens.

The model therefore cannot freely generate arbitrary text while producing the integer.

### Number

For floating-point numbers, the allowed tokens include:

```text
0 1 2 3 4 5 6 7 8 9 .
```

The implementation also tracks the appearance of the decimal point to prevent malformed numbers containing multiple decimal points.

### Boolean

Boolean generation is restricted to the tokens representing:

```text
true
false
```

### String

For strings, generation begins after a quotation mark and continues until another quotation mark is generated.

The generated tokens are then decoded into the final string value.

### Overall pipeline

The complete algorithm can be summarized as:

```text
User prompt
     |
     v
Parse input
     |
     v
Validate function definitions
     |
     v
Construct function-selection prompt
     |
     v
Constrained function-name decoding
     |
     v
Selected function
     |
     v
Read parameter definitions
     |
     v
Constrained parameter-value decoding
     |
     v
Function name + parameter values
     |
     v
Write results
```

---

## Design Decisions

### Constrained decoding instead of free-form generation

The main design decision was to constrain the model's output rather than asking it to generate a complete JSON response.

A free-form model could produce:

```text
The function you probably want is get_weather...
```

or malformed JSON.

With constrained decoding, the implementation directly controls which tokens are valid at each generation step.

This makes the output more predictable and easier to process programmatically.

### Function names are tokenized beforehand

Available function names are encoded during initialization.

This avoids repeatedly encoding the same function names during the selection process and makes prefix matching straightforward.

### Parameter types determine decoding rules

Different parameter types require different constraints.

Instead of implementing one generic extraction algorithm, the project uses specialized extraction methods:

```text
__extract_int_value()
__extract_float_value()
__extract_bool_value()
__extract_str_value()
```

This makes the behavior for each type explicit.

### Pydantic for validation

Pydantic is used to validate input data before it reaches the selection algorithm.

This ensures that malformed function definitions are detected early.

The `FunctionDefinition` model also verifies that parameter and return types belong to the supported set:

```text
number
integer
object
boolean
string
```

### `fn_no_function`

A dedicated fallback function was added instead of forcing the model to select one of the available functions.

This allows requests that do not correspond to any available function to be handled explicitly.

---

## Performance Analysis

### Accuracy

Constrained decoding improves structural reliability because the model cannot generate arbitrary function names.

For function selection, the output is guaranteed to correspond to one of the known function names, assuming the tokenization and matching logic correctly represent those names.

Parameter extraction also benefits from type-specific constraints. For example, an integer extraction cannot freely produce alphabetic text.

However, constrained decoding does **not guarantee semantic accuracy**. The model can still select the wrong function if multiple functions have similar meanings.

Therefore:

```text
Constrained decoding
        ≠
Perfect semantic understanding
```

It mainly improves the **validity and reliability of the generated format**.

### Speed

The main performance cost comes from repeatedly calling:

```python
model.get_logits_from_input_ids(...)
```

during token generation.

For function selection, this happens once per generated token.

Parameter extraction similarly requires multiple model calls for multi-token values.

The implementation prioritizes correctness and deterministic output constraints over minimizing the number of model calls.

### Memory

The implementation stores tokenized function names and uses lists of token IDs during generation.

The additional memory requirements are relatively small compared with the language model itself.

### Reliability

The system validates input files with Pydantic before processing them.

Invalid JSON, invalid schemas, missing fields, and unsupported parameter types are therefore detected before the main selection process.

The constrained token generation also reduces the chance of receiving structurally invalid values.

---

## Challenges Faced

### Restricting model generation

One of the main challenges was preventing the language model from generating invalid function names.

A simple approach would be to ask the model to return a function name and trust the result. However, the model could generate a name that does not exist.

The solution was to tokenize all available function names and restrict every generated token to one that corresponds to a valid function-name prefix.

### Matching token prefixes

Another challenge was determining which tokens are valid after each generated token.

For example:

```text
function_a
function_b
function_c
```

may share the same initial tokens.

The implementation compares the generated sequence with the prefixes of every available function name and uses the remaining tokens as the allowed candidates.

### Extracting different parameter types

Strings, integers, numbers, and booleans have very different valid representations.

The solution was to implement separate extraction strategies and define allowed and stop tokens for numeric and boolean values.

### Input validation

Function definitions contain nested structures, making it possible to receive malformed parameter definitions.

Pydantic models and custom validation were used to ensure that the input follows the expected structure before it is used by the selector.

---

## Testing Strategy

The implementation can be validated at several levels.

### Input validation

Test function-definition files containing:

* Valid function definitions.
* Missing required fields.
* Unknown fields.
* Invalid parameter structures.
* Unsupported parameter types.
* Invalid JSON.

The Pydantic models should reject invalid structures.

### Function selection

Test prompts that clearly correspond to:

* Each available function.
* Multiple similar functions.
* No available function.

For example:

```text
Prompt → matching function
Prompt → matching function
Unrelated prompt → fn_no_function
```

### Parameter extraction

Test each supported parameter type independently:

```text
integer
number
boolean
string
```

Also test:

* Positive integers.
* Negative integers.
* Decimal numbers.
* Negative decimal numbers.
* Boolean values.
* Strings containing spaces.

### End-to-end testing

Run the complete program using a representative input file and verify that:

1. The input files are parsed correctly.
2. Function definitions are validated.
3. The correct function is selected.
4. Parameter values are extracted.
5. The output file is generated.
6. The generated result has the expected structure.

Example result:

```json
[
    {
        "prompt": "Set the temperature to 25 degrees.",
        "name": "set_temperature",
        "parameters": {
            "temperature": 25
        }
    }
]
```

---

## Example Usage

### Basic execution

```bash
uv run python -m src
```

This uses the default files:

```text
data/input/functions_definition.json
data/input/function_calling_tests.json
```

and writes the results to:

```text
data/output/function_calling_results.json
```

### Custom files

```bash
uv run python -m src \
    --functions_definition my_functions.json \
    --input my_prompts.json \
    --output results.json
```

### Example function definition

```json
[
    {
        "name": "set_temperature",
        "description": "Set the temperature.",
        "parameters": {
            "temperature": {
                "type": "number"
            }
        },
        "returns": {
            "type": "boolean"
        }
    }
]
```

### Example prompt

```json
[
    {
        "prompt": "Set the temperature to 21.5 degrees."
    }
]
```

The resulting output is conceptually:

```json
[
    {
        "prompt": "Set the temperature to 21.5 degrees.",
        "name": "set_temperature",
        "parameters": {
            "temperature": 21.5
        }
    }
]
```

---

## Resources

### Python

* **Python Documentation** — Official Python documentation covering the language, standard library, and programming concepts.
* **PEP 8** — Python's official style guide.
* **PEP 257** — Conventions for Python docstrings.

### Pydantic

* **Pydantic Documentation** — Documentation for data validation and model definitions.
* **Pydantic Validators** — Documentation covering model and field validation.

### NumPy

* **NumPy Documentation** — Reference documentation for NumPy and its numerical operations.

### Articles
* **https://www.aidancooper.co.uk/constrained-decoding/**

### AI Usage

AI tools were used as a **learning and development aid** during this project.

AI was used for:

* Understanding Python documentation conventions, particularly **PEP 257**.
* Reviewing and improving project documentation and docstrings.
* Explaining Python, Pydantic, tokenization, and constrained decoding concepts.
* Reviewing implementation ideas and identifying potential bugs or inconsistencies.
* Helping structure this README according to the project's documentation requirements.

---
