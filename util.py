from datetime import datetime, timedelta
import ast
import os
import zipfile
import rfc3339


def extract(folder_path: str):
    if os.path.exists(folder_path):
        extract_directory = "./"
        with zipfile.ZipFile(folder_path, 'r') as zip_ref:
            zip_ref.extractall(extract_directory)
        print(folder_path, "extracted")
    else:
        print(folder_path, "does not exist")


def dict_values_to_list_of_strings(dictionary_string):
    if len(dictionary_string) == 0:
        return []

    dictionary = ast.literal_eval(dictionary_string)
    result = []
    for value in dictionary.values():
        if isinstance(value, list):
            result.extend(str(element) for element in value)
        else:
            result.append(str(value))
    return result


def dict_keys_to_list_of_strings(dictionary_string):
    if len(dictionary_string) == 0:
        return []

    dictionary = ast.literal_eval(dictionary_string)
    result = []
    for key in dictionary.keys():
        if isinstance(key, list):
            result.extend(str(element) for element in key)
        else:
            result.append(str(key))
    return result


def list_string_to_python_list(list_string):
    if len(list_string) == 0:
        return []
    else:
        return ast.literal_eval(list_string)


def to_int(value):
    if isinstance(value, str) and len(value) == 0:
        return 0
    else:
        return int(value)


def to_date(date_str: str):
    if len(date_str) == 0:
        return ""
    else:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        return rfc3339.rfc3339(date)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    if iteration == total:
        print()


def ask_user_choice(prompt_text: str, choices: list):
    print(prompt_text)
    for i, text in enumerate(choices, start=1):
        print(f" {i}) {text}")

    while True:
        chosen_string = input("> Cosa scegli? ")
        try:
            chosen_number = int(chosen_string)
            if 1 <= chosen_number <= len(choices):
                return chosen_number - 1, choices[chosen_number - 1]
        except ValueError:
            pass

        print(f"Inserire un numero da 1 a {len(choices)}!")


def prettify_duration(seconds: int) -> str:
    """
        Turns an integer (number of seconds) to its duration as a prettified string,
        120 -> "2:00"
    :param seconds: timespan to prettify
    :return: timespan in a more human-readable string
    """
    return str(timedelta(seconds=seconds))