from __future__ import annotations


SPEECHBRAIN_TO_SEAMLESS = {
    "en": "eng",
    "am": "amh",
    "fr": "fra",
    "ar": "arb",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "zh": "cmn",
    "ja": "jpn",
    "ko": "kor",
    "tr": "tur",
    "sw": "swh",
}


def speechbrain_to_seamless(
    language: str,
) -> str:

    # SpeechBrain normally returns:
    #
    # "en: English"
    # "am: Amharic"
    #
    # We only need the language code.

    code = language.split(":")[0].strip().lower()

    if code not in SPEECHBRAIN_TO_SEAMLESS:
        raise ValueError(
            f"Unsupported language for SeamlessM4T: "
            f"{language}"
        )

    return SPEECHBRAIN_TO_SEAMLESS[code]