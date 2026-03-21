from src.data_preprocessing import clean_text


def test_clean_text_removes_extra_whitespace_and_emoji():
    raw = "  Hola   mundo 😀\n\t"
    cleaned = clean_text(raw)
    assert cleaned == "Hola mundo"


def test_clean_text_handles_none():
    assert clean_text(None) == ""
