import os.path

import weaviate
import nltk
from transformers import pipeline

from audio_feature_extractor import AudioFeatureExtractor
from util import ask_user_choice, prettify_duration


summarizer_model_name = "facebook/bart-large-cnn"
audio_model_name = "facebook/wav2vec2-base-100k-voxpopuli"
summarizer = None
audio_feature_extractor = None

# Which parameters we want in output
parameters = ["talk_id", "title", "speaker_1", "all_speakers", "occupations", "about_speakers", "views",
              "recorded_date", "published_date", "event", "native_lang", "available_lang", "comments",
              "duration", "topics", "url", "description", "transcript"]  # "related_talks"

def build_query(client: weaviate.Client, limit=3, additional_parameters=None):
    # Which class to look for on the database
    class_name = "TedTalk"

    if additional_parameters is None:
        additional_parameters = ["id", "certainty", "distance"]

    # Performs the query
    query = client.query\
        .get(class_name, parameters)\
        .with_limit(limit)\
        .with_additional(additional_parameters)

    return query


def execute_query(query):
    query = query.do()

    ted_talks = query["data"]["Get"]["TedTalk"]  # navigate the response json and only return the talks
    return ted_talks


def print_result(talk):
    print("============================================")
    print(f"Talk id: {talk['talk_id']}")
    print(f"Title: {talk['title']}")
    print(f"Speaker 1: {talk['speaker_1']} @ {talk['event']}")
    print(f"Native language: {talk['native_lang']}")
    print(f"Duration: {prettify_duration(talk['duration'])}")
    print(f"Description: {talk['description']}")
    print(f"URL: {talk['url']}")
    print("TLDR: ", end="")
    print(summarize(summarizer, talk['transcript']))


def split_large_text_in_segments(long_text, tokenizer):
    # https://discuss.huggingface.co/t/summarization-on-long-documents/920/24
    sentences = nltk.tokenize.sent_tokenize(long_text, language="italian")
    length = 0
    chunk = ""
    chunks = []
    count = -1
    for sentence in sentences:
        count += 1
        combined_length = len(tokenizer.tokenize(sentence)) + length  # add the # of sentence tokens to the length counter

        if combined_length <= tokenizer.max_len_single_sentence:  # if it doesn't exceed
            chunk += sentence + " "  # add the sentence to the chunk
            length = combined_length  # update the length counter

            # if it is the last sentence
            if count == len(sentences) - 1:
                chunks.append(chunk)  # save the chunk

        else:
            chunks.append(chunk)  # save the chunk
            # reset
            length = 0
            chunk = ""

            # take care of the overflow sentence
            chunk += sentence + " "
            length = len(tokenizer.tokenize(sentence))

    return chunks


def summarize(summarizer, long_text):
    # Since this summarizer can't handle texts longer than 1024 characters, we need to split the input text in
    # sentences shorter than 1024. We summarize each sentence and then we join the summarized results

    tokenizer = summarizer.tokenizer
    text_chunks = split_large_text_in_segments(long_text, tokenizer)

    # Performs summarization
    summaries = summarizer(text_chunks, length_penalty=5.0, num_beams=4, max_length=256, early_stopping=True, do_sample=False)

    # Joins the results to get a single text
    summary = ""
    for r in summaries:
        summary += r["summary_text"] + " "

    # Returns the summary
    return summary


def semantic_search(client):
    text = input("> Cosa cerchi? ")
    query = build_query(client, limit=3)
    query = query.with_near_text({
        "concepts": [text]  # Search using a near_text technique
    })
    results = execute_query(query)
    print("Ecco cosa ho trovato:")
    for talk in results:
        print_result(talk)


def hybrid_search(client):
    text = input("> Cosa cerchi? ")
    query = build_query(client, limit=3)
    query = query.with_hybrid(query=text, properties=["transcript"]) #perform hybrid search on transcript only
    results = execute_query(query)
    print("Ecco cosa ho trovato:")
    for talk in results:
        print_result(talk)


def print_qna_result_text(transcript: str, answer: str, answer_start: int, answer_end: int) -> None:
    UNDERLINE = '\033[4m'
    ENDCOLOR = '\033[0m'

    print_start = 0
    print_end = len(transcript)

    if answer_start > 100:
        print_start = answer_start - 100
    if len(transcript) > answer_end + 100:
        print_end = answer_end + 100

    text_before = transcript[print_start:answer_start]
    text_after = transcript[answer_end:print_end]

    print(text_before, end="")
    print(f"{UNDERLINE}{answer}{ENDCOLOR}", end="")
    print(text_after)


def print_qna_result(query_result):
    answer_dict = query_result["_additional"]["answer"]
    if answer_dict["hasAnswer"]:
        answer_text = answer_dict['result']
        certainty = answer_dict['certainty']
        startPosition = answer_dict['startPosition']
        endPosition = answer_dict['endPosition']

        talk_id = query_result['talk_id']
        talk_title = query_result['title']
        talk_transcript = query_result['transcript']

        print(f"# Risposta: {answer_text}")
        print(f"# Certezza: {certainty}")
        print(f"# Estratto dal talk {talk_id}: '{talk_title}'")
        # print_qna_result_text(talk_transcript, answer_text, startPosition, endPosition)
        print("")


def question_and_answer(client):
    user_provided_question = input("> Fai una domanda: ")

    additional_parameters = "answer {hasAnswer certainty property result startPosition endPosition}"
    ask_details = {
        "question": user_provided_question,
        "properties": ["transcript"]
    }

    query = build_query(client, limit=3, additional_parameters=additional_parameters)
    query = query.with_ask(ask_details)  # perform hybrid search on transcript only
    results = execute_query(query)

    answer_found = any([x for x in results if x["_additional"]["answer"]["hasAnswer"]])
    if answer_found:
        print("Ecco cosa ho trovato:")
        for result in results:
            print_qna_result(result)
    else:
        print("Non ho trovato risposte")


def audio_search(client, device):
    global audio_feature_extractor

    if audio_feature_extractor is None:
        print("Initializing audio model...")
        audio_feature_extractor = AudioFeatureExtractor(audio_model_name, device)

    audio_file_path = input("> Audio file path: ")
    if not os.path.exists(audio_file_path):
        print(f"Could not find file {audio_file_path}")
        return

    print("Extracting audio features...")
    audio_features = audio_feature_extractor.extract_long_audio_embedding(audio_file_path)

    print("Querying the database...")
    parameters_string = ' '.join(parameters)
    response = (
        client.query
        .get("TedTalkAudio", [
            "talk_entry { ... on TedTalk { " + parameters_string + " } }"
        ])
        .with_near_vector({
            "vector": audio_features
        })
        .with_limit(3)
        .do()
    )

    ted_talk_audios = response["data"]["Get"]["TedTalkAudio"]
    for ted_talk_audio in ted_talk_audios:
        talk_entry = ted_talk_audio["talk_entry"][0]
        print_result(talk_entry)


if __name__ == '__main__':
    device = 0  # "cpu"
    print("Initializing summarizer model...")
    nltk.download('punkt')
    summarizer = pipeline("summarization", model=summarizer_model_name, device=device)

    print("Connecting to weaviate...")
    client = weaviate.Client("http://localhost:8080")

    while True:
        choices = ["Ricerca semantica", "Ricerca ibrida testuale/semantica", "Question & Answer", "Ricerca audio", "Quit"]
        index, _ = ask_user_choice("Cosa vuoi fare?", choices)

        if index == 0:
            semantic_search(client)
        elif index == 1:
            hybrid_search(client)
        elif index == 2:
            question_and_answer(client)
        elif index == 3:
            audio_search(client, device)
        elif index == 4:
            exit()

