import os
import zipfile


def extract(folder_path: str):
    if os.path.exists(folder_path):
        extract_directory = "./"
        with zipfile.ZipFile(folder_path, 'r') as zip_ref:
            zip_ref.extractall(extract_directory)
        print(folder_path, "extracted")
    else:
        print(folder_path, "does not exist")


def dict_values_to_list_of_strings(dictionary):
    result = []
    for value in dictionary.values():
        if isinstance(value, list):
            result.extend(str(element) for element in value)
        else:
            result.append(str(value))
    return result



# extract("ted_talks_it.zip")

my_dict = {"name": "John", "age": 30, "hobbies": ["reading", "coding", "sports"]}
result_list = dict_values_to_list_of_strings(my_dict)
print(result_list)
