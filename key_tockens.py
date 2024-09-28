from multi_rake import Rake


def get_key_tockens(text: str) -> str:
    """Выделяет ключевые слова из полученной строки

    Args:
        text (str): исходная строка

    Returns:
        str: строка ключевых слов
    """
    rake_model = Rake(max_words=10)
    key_tockens = ""
    for key_t in rake_model.apply(text)[:10]:
        key_tockens += key_t + "; "

    return key_tockens
