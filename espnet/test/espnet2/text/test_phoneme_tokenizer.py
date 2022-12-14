import pytest

from espnet2.text.phoneme_tokenizer import PhonemeTokenizer

params = [None, "g2p_en", "g2p_en_no_space"]
try:
    import pyopenjtalk

    params.extend(
        [
            "pyopenjtalk",
            "pyopenjtalk_accent",
            "pyopenjtalk_kana",
            "pyopenjtalk_accent_with_pause",
            "pyopenjtalk_prosody",
        ]
    )
    del pyopenjtalk
except ImportError:
    pass
try:
    import pypinyin

    params.extend(["pypinyin_g2p", "pypinyin_g2p_phone"])
    del pypinyin
except ImportError:
    pass
try:
    import phonemizer

    params.extend(["espeak_ng_arabic"])
    params.extend(["espeak_ng_german"])
    params.extend(["espeak_ng_french"])
    params.extend(["espeak_ng_spanish"])
    params.extend(["espeak_ng_russian"])
    params.extend(["espeak_ng_greek"])
    params.extend(["espeak_ng_finnish"])
    params.extend(["espeak_ng_hungarian"])
    params.extend(["espeak_ng_dutch"])
    params.extend(["espeak_ng_english_us_vits"])
    params.extend(["espeak_ng_hindi"])
    params.extend(["espeak_ng_italian"])
    params.extend(["espeak_ng_polish"])
    del phonemizer
except ImportError:
    pass
try:
    import g2pk

    params.extend(["g2pk", "g2pk_no_space"])
    del g2pk
except ImportError:
    pass
params.extend(["korean_jaso", "korean_jaso_no_space"])


@pytest.fixture(params=params)
def phoneme_tokenizer(request):
    return PhonemeTokenizer(g2p_type=request.param)


def test_repr(phoneme_tokenizer: PhonemeTokenizer):
    print(phoneme_tokenizer)


@pytest.mark.execution_timeout(5)
def test_text2tokens(phoneme_tokenizer: PhonemeTokenizer):
    if phoneme_tokenizer.g2p_type is None:
        input = "HH AH0 L OW1   W ER1 L D"
        output = ["HH", "AH0", "L", "OW1", " ", "W", "ER1", "L", "D"]
    elif phoneme_tokenizer.g2p_type == "g2p_en":
        input = "Hello World"
        output = ["HH", "AH0", "L", "OW1", " ", "W", "ER1", "L", "D"]
    elif phoneme_tokenizer.g2p_type == "g2p_en_no_space":
        input = "Hello World"
        output = ["HH", "AH0", "L", "OW1", "W", "ER1", "L", "D"]
    elif phoneme_tokenizer.g2p_type == "pyopenjtalk":
        input = "???????????????????????????"
        output = [
            "m",
            "u",
            "k",
            "a",
            "sh",
            "i",
            "w",
            "a",
            "pau",
            "o",
            "r",
            "e",
            "m",
            "o",
            "w",
            "a",
            "k",
            "a",
            "k",
            "a",
            "cl",
            "t",
            "a",
        ]
    elif phoneme_tokenizer.g2p_type == "pyopenjtalk_kana":
        input = "???????????????????????????"
        output = ["???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???", "???"]
    elif phoneme_tokenizer.g2p_type == "pyopenjtalk_accent":
        input = "???????????????????????????"
        output = [
            "m",
            "4",
            "-3",
            "u",
            "4",
            "-3",
            "k",
            "4",
            "-2",
            "a",
            "4",
            "-2",
            "sh",
            "4",
            "-1",
            "i",
            "4",
            "-1",
            "w",
            "4",
            "0",
            "a",
            "4",
            "0",
            "o",
            "3",
            "-2",
            "r",
            "3",
            "-1",
            "e",
            "3",
            "-1",
            "m",
            "3",
            "0",
            "o",
            "3",
            "0",
            "w",
            "2",
            "-1",
            "a",
            "2",
            "-1",
            "k",
            "2",
            "0",
            "a",
            "2",
            "0",
            "k",
            "2",
            "1",
            "a",
            "2",
            "1",
            "cl",
            "2",
            "2",
            "t",
            "2",
            "3",
            "a",
            "2",
            "3",
        ]
    elif phoneme_tokenizer.g2p_type == "pyopenjtalk_accent_with_pause":
        input = "???????????????????????????"
        output = [
            "m",
            "4",
            "-3",
            "u",
            "4",
            "-3",
            "k",
            "4",
            "-2",
            "a",
            "4",
            "-2",
            "sh",
            "4",
            "-1",
            "i",
            "4",
            "-1",
            "w",
            "4",
            "0",
            "a",
            "4",
            "0",
            "pau",
            "o",
            "3",
            "-2",
            "r",
            "3",
            "-1",
            "e",
            "3",
            "-1",
            "m",
            "3",
            "0",
            "o",
            "3",
            "0",
            "w",
            "2",
            "-1",
            "a",
            "2",
            "-1",
            "k",
            "2",
            "0",
            "a",
            "2",
            "0",
            "k",
            "2",
            "1",
            "a",
            "2",
            "1",
            "cl",
            "2",
            "2",
            "t",
            "2",
            "3",
            "a",
            "2",
            "3",
        ]
    elif phoneme_tokenizer.g2p_type == "pyopenjtalk_prosody":
        input = "???????????????????????????"
        output = [
            "^",
            "m",
            "u",
            "[",
            "k",
            "a",
            "sh",
            "i",
            "w",
            "a",
            "_",
            "o",
            "[",
            "r",
            "e",
            "m",
            "o",
            "#",
            "w",
            "a",
            "[",
            "k",
            "a",
            "]",
            "k",
            "a",
            "cl",
            "t",
            "a",
            "$",
        ]
    elif phoneme_tokenizer.g2p_type == "pypinyin_g2p":
        input = "??????????????????????????????"
        output = [
            "ka3",
            "er3",
            "pu3",
            "pei2",
            "wai4",
            "sun1",
            "wan2",
            "hua2",
            "ti1",
            "???",
        ]
    elif phoneme_tokenizer.g2p_type == "pypinyin_g2p_phone":
        input = "??????????????????????????????"
        output = [
            "k",
            "a3",
            "er3",
            "p",
            "u3",
            "p",
            "ei2",
            "uai4",
            "s",
            "uen1",
            "uan2",
            "h",
            "ua2",
            "t",
            "i1",
            "???",
        ]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_arabic":
        input = "???????????? ??????????"
        output = ["??", "a", "s", "s", "a", "l", "??a??", "m", "??", "l", "??i??", "k", "m"]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_german":
        input = "Das h??rt sich gut an."
        output = [
            "d",
            "a",
            "s",
            "h",
            "????",
            "??",
            "t",
            "z",
            "??",
            "??",
            "??",
            "??u??",
            "t",
            "??a",
            "n",
            ".",
        ]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_french":
        input = "Bonjour le monde."
        output = ["b", "????", "??", "??u", "??", "l", "??-", "m", "??????", "d", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_spanish":
        input = "Hola Mundo."
        output = ["??o", "l", "a", "m", "??u", "n", "d", "o", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_russian":
        input = "???????????? ??????."
        output = ["p", "r??", "i", "v??", "??e", "t", "m??", "??i", "r", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_greek":
        input = "???????? ?????? ??????????."
        output = ["j", "??a", "s", "u", "k", "??o", "s", "m", "e", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_finnish":
        input = "Hei maailma."
        output = ["h", "??ei", "m", "??a??", "??", "l", "m", "a", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_hungarian":
        input = "Hell?? Vil??g."
        output = ["h", "????", "l", "l", "o??", "v", "??i", "l", "a??", "??", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_dutch":
        input = "Hallo Wereld."
        output = ["h", "????", "l", "o??", "??", "??????", "r", "??", "l", "t", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_hindi":
        input = "?????????????????? ??????????????????"
        output = ["n", "??", "m", "????", "s", "t", "e??", "d", "????", "n", "??", "j", "??a??"]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_italian":
        input = "Ciao mondo."
        output = ["t??", "??a", "o", "m", "??o", "n", "d", "o", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_polish":
        input = "Witaj ??wiecie."
        output = ["v", "??i", "t", "a", "j", "??", "f??", "????", "t??", "??", "."]
    elif phoneme_tokenizer.g2p_type == "g2pk":
        input = "??????????????? ???????????????."
        output = [
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            " ",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            ".",
        ]
    elif phoneme_tokenizer.g2p_type == "g2pk_no_space":
        input = "??????????????? ???????????????."
        output = [
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            ".",
        ]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_english_us_vits":
        input = "Hello, World."
        output = [
            "h",
            "??",
            "l",
            "??",
            "o",
            "??",
            ",",
            "<space>",
            "w",
            "??",
            "??",
            "??",
            "l",
            "d",
            ".",
        ]
    elif phoneme_tokenizer.g2p_type == "korean_jaso":
        input = "?????? ????????? ?????????."
        output = [
            "???",
            "???",
            "???",
            "???",
            "???",
            "<space>",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "<space>",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            ".",
        ]
    elif phoneme_tokenizer.g2p_type == "korean_jaso_no_space":
        input = "?????? ????????? ?????????."
        output = [
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            ".",
        ]
    elif phoneme_tokenizer.g2p_type == "is_g2p":
        input = "hlaupa ?? burtu ?? dag"
        output = [
            "l_0",
            "9i:",
            ".",
            "p",
            "a",
            ",",
            "i:",
            ",",
            "p",
            "Y",
            "r_0",
            ".",
            "t",
            "Y",
            ",",
            "i:",
            ",",
            "t",
            "a:",
            "G",
        ]
    else:
        raise NotImplementedError
    assert phoneme_tokenizer.text2tokens(input) == output


def test_token2text(phoneme_tokenizer: PhonemeTokenizer):
    assert phoneme_tokenizer.tokens2text(["a", "b", "c"]) == "abc"
