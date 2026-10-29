"""Make Russian text fully sayable by Silero v5_5_ru.

Silero's Russian model silently drops every character outside its Cyrillic alphabet: Latin
words, digits and most symbols vanish from the audio. A Russian message that names "staging",
"Airwallex" or "PR #41" therefore loses exactly the words that carry its meaning. This module
rewrites those runs into a Cyrillic reading - how a Russian speaker would say them - so nothing
reaches Silero that it would skip. The result never contains an ASCII Latin letter or a digit.
"""

from __future__ import annotations

import re
import unicodedata

# Words a letter-by-letter rule reads badly; keys are lowercase.
_LEXICON: dict[str, str] = {
    "agent": "эйджент",
    "airwallex": "эйрволлекс",
    "android": "андроид",
    "api": "эй-пи-ай",
    "backend": "бэкенд",
    "branch": "бранч",
    "bug": "баг",
    "build": "билд",
    "chatgpt": "чат-джи-пи-ти",
    "claude": "клод",
    "cloud": "клауд",
    "codex": "кодекс",
    "commit": "коммит",
    "dashboard": "дашборд",
    "deploy": "деплой",
    "docker": "докер",
    "email": "имейл",
    "fix": "фикс",
    "frontend": "фронтенд",
    "git": "гит",
    "github": "гитхаб",
    "gitlab": "гитлаб",
    "google": "гугл",
    "id": "ай-ди",
    "iphone": "айфон",
    "k3s": "кей-три-эс",
    "k8s": "кей-эйт-эс",
    "kubernetes": "кубернетис",
    "linux": "линукс",
    "main": "мейн",
    "merge": "мёрдж",
    "ok": "окей",
    "okay": "окей",
    "openai": "оупен-эй-ай",
    "production": "продакшн",
    "pull": "пул",
    "push": "пуш",
    "python": "пайтон",
    "release": "релиз",
    "request": "реквест",
    "readme": "ридми",
    "review": "ревью",
    "sean": "шон",
    "server": "сервер",
    "staging": "стейджинг",
    "telegram": "телеграм",
    "the": "зе",
    "typescript": "тайпскрипт",
    "javascript": "джаваскрипт",
    "user": "юзер",
    "view": "вью",
    "visa": "виза",
    "windows": "виндоус",
    "www": "дабл-ю дабл-ю дабл-ю",
}

# English letter names, for acronyms and lone letters.
_LETTER_NAMES: dict[str, str] = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи",
    "h": "эйч", "i": "ай", "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн",
    "o": "оу", "p": "пи", "q": "кью", "r": "ар", "s": "эс", "t": "ти", "u": "ю",
    "v": "ви", "w": "дабл-ю", "x": "экс", "y": "уай", "z": "зед",
}

_SINGLE_LETTERS: dict[str, str] = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х",
    "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
    "y": "и", "z": "з",
}

# Longest first; matched greedily left to right.
_GRAPHEMES: tuple[tuple[str, str], ...] = (
    ("tion", "шн"), ("sion", "жн"), ("ight", "айт"),
    ("igh", "ай"), ("tch", "ч"), ("dge", "дж"), ("sch", "ск"),
    ("sh", "ш"), ("ch", "ч"), ("th", "т"), ("ph", "ф"), ("ck", "к"), ("qu", "кв"),
    ("ee", "и"), ("oo", "у"), ("ea", "и"), ("ai", "ей"), ("ay", "ей"), ("ey", "ей"),
    ("oi", "ой"), ("oy", "ой"), ("ou", "ау"), ("ow", "оу"), ("oa", "оу"), ("au", "о"),
    ("aw", "о"), ("ew", "ью"), ("wh", "у"),
)

# Latin letters NFKD cannot reduce to ASCII; keys are lowercase.
_LATIN_SPECIALS: dict[str, str] = {
    "ß": "ss", "ẞ": "ss", "œ": "oe", "æ": "ae", "ø": "o", "đ": "d", "ð": "d", "ł": "l",
    "þ": "th", "ı": "i", "ŋ": "n", "ħ": "h", "ŧ": "t",
}
_VOWELS = frozenset("aeiouy")
_MAGIC_E_VOWELS = {"a": "ей", "i": "ай", "o": "оу", "u": "ю"}

_UNITS = ("ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
_UNITS_FEMININE = ("ноль", "одна", "две") + _UNITS[3:]
_TEENS = (
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
    "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
)
_TENS = (
    "", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
    "восемьдесят", "девяносто",
)
_HUNDREDS = (
    "", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот",
    "восемьсот", "девятьсот",
)
_SCALES = (
    ("тысяча", "тысячи", "тысяч"),
    ("миллион", "миллиона", "миллионов"),
    ("миллиард", "миллиарда", "миллиардов"),
)
_MAX_SPOKEN_DIGITS = 12

_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"'`)\]]+", re.IGNORECASE)
# Starts only at a token start and never backtracks, so a long token without "@" stays linear.
_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]++@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
# A Latin/digit token, possibly joined by in-word separators (file.py, state-eld, snake_case).
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._'’-][A-Za-z0-9]+)*")
_RUN = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
_LATIN = re.compile(r"[A-Za-z]")
_LATIN_NAME_LETTER = re.compile(
    r"(?:LETTER|LIGATURE)(?: SMALL| CAPITAL)* ([A-Z]{1,2})(?: WITH .*)?$"
)


def speakable_russian_text(text: str) -> str:
    """Rewrite Latin words, digits and symbols in ``text`` into a Cyrillic reading."""
    text = _fold_foreign_characters(text.replace("№", " номер "))
    text = _URL.sub(lambda match: _speak_url(match.group(0)), text)
    text = _EMAIL.sub(lambda match: _speak_email(match.group(0)), text)
    text = re.sub(r"#\s*(?=\d)", " номер ", text)
    text = re.sub(r"(?<=[A-Za-z])#(?!\w)", " шарп", text)
    text = _TOKEN.sub(_speak_token_apart, text)
    text = text.replace("%", " процентов").replace("&", " и ").replace("+", " плюс ")
    text = text.replace("@", " эт ")
    # Unreachable by construction; a guard so no Latin letter can ever reach Silero.
    text = _LATIN.sub(lambda match: _SINGLE_LETTERS[match.group(0).lower()], text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"[ \t]([,.;:!?])", r"\1", text).strip()


def _fold_foreign_characters(text: str) -> str:
    """Fold accented, fullwidth, styled and superscript Latin letters and digits to ASCII."""
    folded: list[str] = []
    for character in text:
        if character.isascii():
            folded.append(character)
            continue
        digit = unicodedata.digit(character, None)
        if digit is not None:
            folded.append(str(digit))
            continue
        compatible = unicodedata.normalize("NFKC", character)
        if compatible != character and compatible.isascii() and compatible.isalnum():
            folded.append(compatible)
            continue
        name = unicodedata.name(character, "")
        if "LATIN" not in name:
            folded.append(character)
            continue
        base = "".join(
            part
            for part in unicodedata.normalize("NFKD", character)
            if not unicodedata.combining(part)
        )
        if base.isascii() and base.isalpha():
            folded.append(base)
        elif character.lower() in _LATIN_SPECIALS:
            folded.append(_LATIN_SPECIALS[character.lower()])
        else:
            # "LATIN LETTER SMALL CAPITAL F", "LATIN CAPITAL LETTER Z WITH STROKE": the letter
            # is named after LETTER and its size words.
            letter = _LATIN_NAME_LETTER.search(name)
            folded.append(letter.group(1) if letter else "")
    return "".join(folded)


def _speak_url(url: str) -> str:
    stripped = url.rstrip(".,;:!?")
    trailing = url[len(stripped):]
    url = re.sub(r"^https?://", "", stripped, flags=re.IGNORECASE).rstrip("/?#")
    spoken = url.replace("/", " слэш ")
    return _speak_separated(spoken) + trailing


def _speak_email(address: str) -> str:
    local, _, domain = address.partition("@")
    return f"{_speak_separated(local)} эт {_speak_separated(domain)}"


def _speak_separated(text: str) -> str:
    text = re.sub(r"(?<=\w)\.(?=\w)", " точка ", text)
    text = re.sub(r"[_=&:~?#]", " ", text)
    return _TOKEN.sub(_speak_token_apart, text)


def _speak_token_apart(match: re.Match[str]) -> str:
    """Speak a token, kept apart from a Cyrillic letter it touches ("м2" -> "м два")."""
    source = match.string
    before = " " if match.start() > 0 and source[match.start() - 1].isalpha() else ""
    after = " " if match.end() < len(source) and source[match.end()].isalpha() else ""
    return before + _speak_token(match.group(0)) + after


def _speak_token(token: str) -> str:
    lexical = _LEXICON.get(token.lower())
    if lexical is not None:
        return lexical
    pieces = re.split(r"([._'’-])", token)
    spoken: list[str] = []
    for index, piece in enumerate(pieces):
        if piece == ".":
            spoken.append(" точка ")
        elif piece == "-" and pieces[index - 1].isdigit() and pieces[index + 1].isdigit():
            # A range or a difference ("10-20"): Russian reads it "десять-двадцать".
            spoken.append("-")
        elif piece in {"_", "-"}:
            spoken.append(" ")
        elif piece in {"'", "’"}:
            continue
        elif piece:
            spoken.append(_speak_piece(piece))
    return "".join(spoken)


def _speak_piece(piece: str) -> str:
    lexical = _LEXICON.get(piece.lower())
    if lexical is not None:
        return lexical
    return " ".join(_speak_run(run) for run in _RUN.findall(piece))


def _speak_run(run: str) -> str:
    if run.isdigit():
        return _speak_number(run)
    lexical = _LEXICON.get(run.lower())
    if lexical is not None:
        return lexical
    lowered = run.lower()
    if len(run) == 1 or (run.isupper() and len(run) <= 5) or not (_VOWELS & set(lowered)):
        return "-".join(_LETTER_NAMES[letter] for letter in lowered)
    return _transliterate(lowered)


def _transliterate(word: str) -> str:
    magic_vowel = ""
    if (
        len(word) > 3
        and word.endswith("e")
        and word[-2] not in _VOWELS
        and word[-3] in _MAGIC_E_VOWELS
        and word[-4] not in _VOWELS
    ):
        magic_vowel = _MAGIC_E_VOWELS[word[-3]]
        word = word[:-3] + "\0" + word[-2]
    elif len(word) > 3 and word.endswith("e") and word[-2] not in _VOWELS:
        word = word[:-1]
    output: list[str] = []
    index = 0
    while index < len(word):
        if word[index] == "\0":
            output.append(magic_vowel)
            index += 1
            continue
        for grapheme, reading in _GRAPHEMES:
            if word.startswith(grapheme, index):
                output.append(reading)
                index += len(grapheme)
                break
        else:
            letter = word[index]
            following = word[index + 1] if index + 1 < len(word) else ""
            if letter == "c" and following in {"e", "i", "y"}:
                output.append("с")
            elif letter == "y" and index == 0 and following in _VOWELS:
                output.append("й")
            elif letter == "e" and index == 0:
                output.append("э")
            else:
                output.append(_SINGLE_LETTERS[letter])
            index += 1
    return "".join(output)


def _speak_number(digits: str) -> str:
    if len(digits) > _MAX_SPOKEN_DIGITS or (len(digits) > 1 and digits.startswith("0")):
        return " ".join(_UNITS[int(digit)] for digit in digits)
    value = int(digits)
    if value == 0:
        return _UNITS[0]
    words: list[str] = []
    groups: list[int] = []
    while value:
        groups.append(value % 1000)
        value //= 1000
    for scale_index in range(len(groups) - 1, -1, -1):
        group = groups[scale_index]
        if group == 0:
            continue
        words.extend(_speak_triplet(group, feminine=scale_index == 1))
        if scale_index > 0:
            words.append(_plural_form(group, _SCALES[scale_index - 1]))
    return " ".join(words)


def _speak_triplet(value: int, *, feminine: bool) -> list[str]:
    words: list[str] = []
    hundreds, rest = divmod(value, 100)
    if hundreds:
        words.append(_HUNDREDS[hundreds])
    if 10 <= rest < 20:
        words.append(_TEENS[rest - 10])
        return words
    tens, units = divmod(rest, 10)
    if tens:
        words.append(_TENS[tens])
    if units:
        words.append((_UNITS_FEMININE if feminine else _UNITS)[units])
    return words


def _plural_form(value: int, forms: tuple[str, str, str]) -> str:
    if 11 <= value % 100 <= 14:
        return forms[2]
    if value % 10 == 1:
        return forms[0]
    if 2 <= value % 10 <= 4:
        return forms[1]
    return forms[2]
