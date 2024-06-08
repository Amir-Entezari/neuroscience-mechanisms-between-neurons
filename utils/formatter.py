import json


def pretty_format_dict(data):
    # Convert the dictionary to a JSON string with an indent
    json_string = json.dumps(data, indent=4)
    # Split the string into lines
    lines = json_string.splitlines()
    # Remove the first and last line to strip the outer braces
    lines = lines[1:-1]
    # Remove the first four spaces from each line
    lines = [line[4:] for line in lines]
    # Join the lines back into a single string
    return '\n'.join(lines).replace("\n},\n", "},\n").replace("\"", "")