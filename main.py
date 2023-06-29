import datetime

import torch
import weaviate
import rfc3339
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.tokenize import sent_tokenize

summarizer_model_name = "facebook/bart-large-cnn"
summarizer = None


def search(client, text, limit=3):

    # Which class to look for on the database
    class_name = "TedTalk"

    # Which parameters we want in output
    parameters = ["talk_id", "title", "speaker_1", "all_speakers", "occupations", "about_speakers", "views",
                  "recorded_date", "published_date", "event", "native_lang", "available_lang", "comments",
                  "duration", "topics", "url", "description", "transcript"] # "related_talks"

    # Additional parameters set by the dbms
    additional_parameters = ["id", "certainty", "distance"]

    # Performs the query
    query_result = client.query\
        .get(class_name, parameters)\
        .with_near_text({
            "concepts": [text] # Serach using a near_text technique
        })\
        .with_limit(limit)\
        .with_additional(additional_parameters)\
        .do()

    ted_talks = query_result["data"]["Get"]["TedTalk"]  # navigate the response json and only return the talks
    return ted_talks


def prettify_duration(seconds: int) -> str:
    """
        Turns an integer (number of seconds) to its duration as a prettified string,
        120 -> "2:00"
    :param seconds: timespan to prettify
    :return: timespan in a more human-readable string
    """
    return str(datetime.timedelta(seconds=seconds))


def print_result(talk):
    print("============================================")
    print(f"Talk id: {talk['talk_id']}")
    print(f"Title: {talk['title']}")
    print(f"Speaker 1: {talk['speaker_1']} @ {talk['event']}")
    print(f"Native language: {talk['native_lang']}")
    print(f"Duration: {prettify_duration(talk['duration'])}")
    print(f"Description: {talk['description']}")
    print(f"URL: {talk['url']}")
    print(f"TLDR: {summarize(summarizer, talk['transcript'])}")
    pass


def split_large_text_in_segments(long_text, tokenizer):
    # https://discuss.huggingface.co/t/summarization-on-long-documents/920/24
    sentences = nltk.tokenize.sent_tokenize(long_text)
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


if __name__ == '__main__':
    print("Initializing summarizer model...")
    nltk.download('punkt')
    summarizer = pipeline("summarization", model=summarizer_model_name, device=0)

    print("Connecting to weaviate...")
    client = weaviate.Client("http://localhost:8080")

    while True:
        # Asks user to provide a search prompt. If empty, the application terminates
        text = input("> Cosa cerchi? ")
        if len(text) == 0:
            break

        # Search the database. Limits the maximum number of returner talks
        query_result = search(client, text, limit=3)

        # Prints the talks
        print("Ecco cosa ho trovato:")
        for talk in query_result:
            print_result(talk)
