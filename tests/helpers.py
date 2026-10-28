"""Test helpers: a byte-level stand-in tokenizer so tests don't need to download GPT-2 files."""


class ByteEncoder:
    """Same interface as the parts of tiktoken we use. NOT used for real training."""

    n_vocab = 256
    eot_token = 0

    def encode_ordinary(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")
