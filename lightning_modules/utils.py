import re
from typing import Callable
from unicodedata import normalize

from constants import MODEL_NAME
from transformers import AutoTokenizer, BertTokenizerFast

url_pattern = re.compile(r"(https?://[^\s]+)")
mention_pattern = re.compile(r"@[^\s]+")
pattern1 = re.compile(
    r"(_[a-zA-Z0-9]+)|(#[\u0900-\u097F]+)|(@[\u0900-\u097F]+)|(_[\u0900-\u097F]+)"
)
pattern2 = re.compile(r"(\W)(?=\1)")
multi_whitespaces = re.compile(r"\s+")

to_replace = """@#=/+…:"")(}{][*%_’‘'"""


def cleaner(text: str) -> str:
    """
    Cleans the tweets using the fllowing sequential steps:
        - Remove all the words starting with '_' ("_ahiraj"),
            mentions starting with '_' ("@_Silent__Eyes__") and also
            Devanagiri ("@पाेखरा") and hashtags used with Devanagiri ("#पाेखरा").
        - Remove punctuations (selected manually).
            This does not include sentence enders like "|" or "." or "?" or "!".
        - Removes bad characters like "&gt;".
        - If a punctuation or a whitespace has been repeated multiple times,
            adjust it to a single occurence.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    no_url = re.sub(url_pattern, "", text)
    no_mention = re.sub(mention_pattern, "", no_url)
    new_text = (
        re.sub(pattern1, "", no_mention)
        .translate(str.maketrans("", "", to_replace))
        .translate(str.maketrans("", "", "&gt;"))
    )
    remove_repitition = re.sub(multi_whitespaces, " ", re.sub(pattern2, "", new_text))
    final_text = normalize("NFKC", remove_repitition).strip()
    return final_text


def tokenizer_normalize(model_name: str = MODEL_NAME) -> Callable[[str], str]:
    """
    Returns a function that normalizes the text using the tokenizer's
    """

    tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained(model_name)

    return tokenizer.backend_tokenizer.normalizer.normalize_str
