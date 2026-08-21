import json


def run(payload):
    text = payload.get("text", "")
    if not text:
        return "No text provided. Use {\"text\": \"your text here\"}"

    words = text.split()
    char_count = len(text)
    word_count = len(words)
    line_count = text.count("\n") + 1

    unique_words = len(set(w.lower() for w in words))

    return json.dumps({
        "characters": char_count,
        "words": word_count,
        "lines": line_count,
        "unique_words": unique_words,
    }, indent=2)
