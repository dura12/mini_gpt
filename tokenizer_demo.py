from tinygpt.data import get_tokenizer

enc = get_tokenizer()
sentences = [
    "The cat sat on the mat.",
    "Addis Ababa is the capital of Ethiopia.",
    "Language models predict the next token.",
    "Unbelievably, tokenisation splits rare words into pieces.",
    "In 2026, iCog Labs trained a tiny GPT!",
]

for s in sentences:
    ids = enc.encode_ordinary(s)
    pieces = [enc.decode([i]) for i in ids]
    print("text   :", s)
    print("pieces :", pieces)
    print("ids    :", ids)
    print("decoded:", enc.decode(ids), "\n")
