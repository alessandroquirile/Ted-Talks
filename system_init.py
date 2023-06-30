import os
import pandas as pd
import weaviate
from weaviate.util import generate_uuid5
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from audio_processing import extract_long_audio_embedding
from util import *


ted_talks_csv_path = "dataset/ted_talks_it.csv"
ted_talks_zip_path = "dataset/ted_talks_it.zip"
ted_talks_audio_path = "dataset/AUDIO/"

audio_model_name = "facebook/wav2vec2-large-xlsr-53"

TedTalkClassName = "TedTalk"
TedTalkAudioClassName = "TedTalkAudio"

def is_database_already_configured():
    print("Getting the schema...")
    schema = client.schema.get()  # Get the schema to test connection
    class_already_exists = any(x for x in schema["classes"] if x["class"] == "TedTalk")
    return class_already_exists


def create_schema(ted_talk_object_schema):
    print(f"Creating database schema...")
    client.schema.create(ted_talk_object_schema)



def ask_for_similarity_metric():
    # Prints the available metrics
    print("Meriche disponibili:")
    metrics = ["cosine", "dot", "l2", "hamming", "manhattan"]
    for i, metric in enumerate(metrics, start=1):
        print(f" {i}) {metric}")

    # Asks the user for a metric util a correct number is provided
    while True:
        chosen_string = input("> Quale metrica di similarità usare?")
        try:
            chosen_number = int(chosen_string)
            if 1 <= chosen_number <= len(metrics):
                return metrics[chosen_number - 1]
        except ValueError:
            pass

        print(f"Inserire un numero da 1 a {len(metrics)}!")


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
            "class": TedTalkClassName,
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
                "distance": "cosine",
            }
        },
        {
            "class": TedTalkAudioClassName,
            "description": "Stores the audio embedding for a Ted Talk",
            "properties": [
                {
                    "name": "talk_entry",
                    "description": "Cross reference to the related talk",
                    "dataType": ["TedTalk"]
                },
                {
                    "name": "original_file_name",
                    "description": "The MP3 file this embedding was extracted from",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                },
            ],
            "vectorIndexConfig": {
                "distance": "cosine",
            }
        }
    ]
}


def check_dataset_files():
    if not os.path.exists(ted_talks_csv_path):
        if os.path.exists(ted_talks_zip_path):
            print("Extracting zip file...")
            extract("ted_talks_it.zip")
        else:
            print("Could not find dataset files. Quitting.")
            exit(-1)

    if not os.path.exists(ted_talks_audio_path):
        print("Could not find AUDIO files. Quitting.")
        exit(-1)


def store_ted_talks(batch, ted_talks, id_to_uuid):
    print("Storing objects...")
    for index, talk in enumerate(ted_talks):
        print_progress_bar(index, len(ted_talks))
        batch.add_data_object(
            data_object=talk,
            class_name="TedTalk",
            uuid=id_to_uuid[talk["talk_id"]]
        )


def store_ted_talks_relations(batch, ted_talks_it_dataframe, id_to_uuid):
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
                    from_object_class_name=TedTalkClassName,
                    from_property_name="related_talks",
                    to_object_uuid=related_talk_uuid,
                    to_object_class_name=TedTalkClassName,
                )
            else:
                print(f"Talk id {talk_id} not found in this dataset. Reference dropped")


def prepare_objects(ted_talks_it_dataframe, id_to_uuid, ted_talks):
    print("Preparing data objects...")
    for ix, row in ted_talks_it_dataframe.iterrows():
        talk_object = build_talk_object(row)

        talk_uuid = generate_uuid5(talk_object, TedTalkClassName)
        id_to_uuid[talk_object["talk_id"]] = talk_uuid
        ted_talks.append(talk_object)


def store_talk_audio_embeddings(batch, ted_talks_it_dataframe, id_to_uuid, feature_extractor, audio_model, device):
    print("Preparing and storing audio embeddings...")
    for index, row in ted_talks_it_dataframe.iterrows():
        file_name = f"{row.talk_id}.mp3"
        audio_file_path = ted_talks_audio_path + file_name

        # check if the mp3 file exists
        if not os.path.exists(audio_file_path):
            print(f"Audio file {audio_file_path} not found: ignoring this entry")
            continue

        # print progress so far
        print_progress_bar(index, len(ted_talks_it_dataframe))

        # if the mp3 file exists, then extract its features and add them to the database
        file_features = extract_long_audio_embedding(audio_file_path, feature_extractor, audio_model, device=device)
        talk_uuid = id_to_uuid[row.talk_id]

        # construct Talk Audio Object
        talk_audio_object = {
            "original_file_name": file_name
        }

        # generate object's uuid
        talk_audio_uuid = generate_uuid5(talk_audio_object, TedTalkAudioClassName)

        # store object
        batch.add_data_object(
            data_object=talk_audio_object,
            class_name=TedTalkAudioClassName,
            uuid=talk_audio_uuid,
            vector=file_features
        )

        # add a reference from this audio to the related TedTalk object
        batch.add_reference(
            from_object_uuid=talk_audio_uuid,
            from_object_class_name=TedTalkAudioClassName,
            from_property_name="talk_entry",
            to_object_uuid=talk_uuid,
            to_object_class_name=TedTalkClassName,
        )


def check_if_database_is_already_configured(client):
    if is_database_already_configured():
        print("Weaviate is already configured!")
        print("Continuing will DELETE the existing TedTalk schema")
        user_input = input("Type 'Yes' to continue, 'No' to quit")

        user_wants_to_continue = (user_input == 'Yes')
        if user_wants_to_continue:
            # Delete the schema to reset the system:
            print("Deleting the existing TedTalk schema")
            client.schema.delete_class(TedTalkClassName)
            client.schema.delete_class(TedTalkAudioClassName)
        else:
            print("Quitting with no changes.")
            exit()


if __name__ == '__main__':
    device = 0  # or "cpu"
    print(f"Loading model {audio_model_name}...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name, device=device)
    audio_model = Wav2Vec2Model.from_pretrained(audio_model_name).to(device)

    check_dataset_files()

    print("Connecting to weaviate...")
    client = weaviate.Client("http://localhost:8080")
    client.batch.configure(
        batch_size=1,
        dynamic=True,
        num_workers=16,
    )

    check_if_database_is_already_configured(client)

    metric = ask_for_similarity_metric()
    for class_ in ted_talk_object_schema["classes"]:
        class_["vectorIndexConfig"]["distance"] = metric

    print("Reading CSV...")
    ted_talks_it_dataframe = pd.read_csv(ted_talks_csv_path).fillna(value="")
    ted_talks = []
    id_to_uuid = {}

    create_schema(ted_talk_object_schema)
    prepare_objects(ted_talks_it_dataframe, id_to_uuid, ted_talks)

    with client.batch as batch:
        store_ted_talks(batch, ted_talks, id_to_uuid)
        store_ted_talks_relations(batch, ted_talks_it_dataframe, id_to_uuid)
        store_talk_audio_embeddings(batch, ted_talks_it_dataframe, id_to_uuid, feature_extractor, audio_model, device)

    print("Task completed.")
