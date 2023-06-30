import ast
import os
import zipfile
from datetime import datetime

import pandas as pd
import rfc3339
import weaviate
from weaviate.util import generate_uuid5


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


def database_already_configured():
    print("Getting the schema...")
    schema = client.schema.get()  # Get the schema to test connection
    class_already_exists = any(x for x in schema["classes"] if x["class"] == "TedTalk")
    return class_already_exists


def create_schema(ted_talk_object_schema):
    print("Creating TedTalk schema...")
    client.schema.create(ted_talk_object_schema)


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


def ask_similarity_metric():
    # Prints the available metrics
    print("Meriche disponibili:")
    metrics = ["cosine", "dot", "l2", "hamming", "manhattan"]
    for i, metric in enumerate(metrics, start=1):
        print(f" {i}) {metric}")

    # Asks the user for a metric util a correct number is provided
    while True:
        chosen_string = input("Quale metrica di similarità usare?")
        try:
            chosen_number = int(chosen_string)
            if 1 <= chosen_number <= len(metrics):
                return metrics[chosen_number - 1]
        except ValueError:
            pass

        print(f"Inserire un numero da 1 a {len(metrics)}")


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    # Credit: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
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
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def build_talk_object(row):
    return {
        "talk_id": row.talk_id,
        "title": row.title,
        "speaker_1": row.speaker_1,
        "all_speakers": dict_values_to_list_of_strings(row.all_speakers),
        "occupations": dict_values_to_list_of_strings(row.occupations),
        "about_speakers": dict_values_to_list_of_strings(row.about_speakers),
        "views": to_int(row.views),
        "recorded_date": to_date(row.recorded_date),
        "published_date": to_date(row.published_date),
        "event": row.event,
        "native_lang": row.native_lang,
        "available_lang": list_string_to_python_list(row.available_lang),
        "comments": to_int(row.comments),
        "duration": to_int(row.duration),
        "topics": list_string_to_python_list(row.topics),
        # related_talks is set manually to add talk references
        "url": row.url,
        "description": row.description,
        "transcript": row.transcript
    }


ted_talk_object_schema = {
    "classes": [
        {
            "class": "TedTalk",
            "description": "Information about a Ted talk",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {
                    "name": "talk_id",
                    "description": "CSV Talk identifier",
                    "dataType": ["int"]
                },
                {
                    "name": "title",
                    "description": "Localized Talk title",
                    "dataType": ["text"]
                },
                {
                    "name": "speaker_1",
                    "description": "The name of the first speaker in this talk",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "all_speakers",
                    "description": "Names of people who spoke in this talk (List)",
                    "dataType": ["text[]"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "occupations",
                    "description": "Occupations of people who spoke in this talk (List)",
                    "dataType": ["text[]"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "about_speakers",
                    "description": "Information about the people who spoke in this talk (List)",
                    "dataType": ["text[]"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "views",
                    "description": "Talk's view count",
                    "dataType": ["int"]
                },
                {
                    "name": "recorded_date",
                    "description": "Talk's recorded date",
                    "dataType": ["date"]
                },
                {
                    "name": "published_date",
                    "description": "Talk's published date",
                    "dataType": ["date"]
                },
                {
                    "name": "event",
                    "description": "Name of the event in which this Ted talk was held",
                    "dataType": ["string"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "native_lang",
                    "description": "Talk's native language",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "available_lang",
                    "description": "List of languages this ted talk is available into",
                    "dataType": ["text[]"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "comments",
                    "description": "Number of comments in this talk",
                    "dataType": ["int"]
                },
                {
                    "name": "duration",
                    "description": "Talk's duration in seconds",
                    "dataType": ["int"]
                },
                {
                    "name": "topics",
                    "description": "Talk's topics",
                    "dataType": ["text[]"]
                },
                {
                    "name": "related_talks",
                    "description": "Cross reference to related talks",
                    "dataType": ["TedTalk"]
                },
                {
                    "name": "url",
                    "description": "URL to this talk",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
                {
                    "name": "description",
                    "description": "Talk's description",
                    "dataType": ["text"]
                },
                {
                    "name": "transcript",
                    "description": "Talk's transcription",
                    "dataType": ["text"]
                }

            ],
            "vectorIndexConfig": {
                "distance": "dot",
            }
        }
    ]
}

if __name__ == '__main__':
    ted_talks_csv_path = "ted_talks_it.csv"

    if not os.path.exists(ted_talks_csv_path):
        print("Extracting zip file...")
        extract("ted_talks_it.zip")

    print("Connecting to weaviate...")
    client = weaviate.Client("http://localhost:8080")
    client.batch.configure(
        batch_size=1,
        dynamic=True,
        num_workers=16,
    )

    if database_already_configured():
        print("Weaviate is already configured!")
        print("Continuing will DELETE the existing TedTalk schema")
        user_input = input("Type 'Yes' to continue, 'No' to quit")

        user_wants_to_continue = (input == 'Yes')
        if user_wants_to_continue:
            # Delete the schema to reset the system:
            print("Deleting the existing TedTalk schema")
            client.schema.delete_class("TedTalk")
        else:
            print("Quitting with no changes.")
            exit()

    metric = ask_similarity_metric()
    for class_ in ted_talk_object_schema["classes"]:
        class_["vectorIndexConfig"]["distance"] = metric

    print("Reading CSV...")
    ted_talks_it_dataframe = pd.read_csv(ted_talks_csv_path).fillna(value="")
    ted_talks = []
    id_to_uuid = {}

    print("Creating schema...")
    create_schema(ted_talk_object_schema)

    print("Preparing data objects...")
    for ix, row in ted_talks_it_dataframe.iterrows():
        talk_object = build_talk_object(row)

        talk_uuid = generate_uuid5(talk_object, "TedTalk")
        id_to_uuid[talk_object["talk_id"]] = talk_uuid
        ted_talks.append(talk_object)

    with client.batch as batch:
        print("Creating objects...")
        for index, talk in enumerate(ted_talks):
            print_progress_bar(index, len(ted_talks))
            batch.add_data_object(
                data_object=talk,
                class_name="TedTalk",
                uuid=id_to_uuid[talk["talk_id"]]
            )

        print("Creating object references...")
        for index, row in ted_talks_it_dataframe.iterrows():
            print_progress_bar(index, len(ted_talks_it_dataframe))
            this_talk_id = id_to_uuid[row.talk_id]
            related_talks_ids = dict_keys_to_list_of_strings(row.related_talks)

            for talk_id in related_talks_ids:
                talk_id = to_int(talk_id)
                if talk_id in id_to_uuid:
                    related_talk_uuid = id_to_uuid[talk_id]
                    batch.add_reference(
                        from_object_uuid=this_talk_id,
                        from_object_class_name="TedTalk",
                        from_property_name="related_talks",
                        to_object_uuid=related_talk_uuid,
                        to_object_class_name="TedTalk",
                    )
                else:
                    print(f"Talk id {talk_id} not found in this dataset. Reference dropped")

    print("Data loaded")
