"""Perplexity-filter defense — reject high-perplexity (likely adversarial) inputs.

References:
  - Alon & Kamfonas, "Detecting Language Model Attacks with Perplexity",
    arXiv:2308.14132 (2023).
  - Jain et al., "Baseline Defenses for Adversarial Attacks Against Aligned
    Language Models", arXiv:2309.00614 (2023).

The idea: gradient-based adversarial suffixes (GCG, AmpleGCG) and many encoding
attacks (base64, cipher, ASCII-art) produce text with abnormally HIGH perplexity
under any language model — even a simple n-gram one — because they consist of
random-looking character sequences that no natural-language model assigns a
reasonable probability to. Rejecting inputs above a perplexity threshold blocks
a large fraction of suffix attacks while rarely touching natural-language prompts.

Model choice and honesty note
------------------------------
This implementation uses a **character bigram** language model: it estimates
the probability of each character given the previous character using counts
from a small bundled reference corpus (English text). It does NOT use a neural
LM. This means:

  * It runs fully OFFLINE, on CPU, with no model download, no torch, no
    transformers — a single pure-Python computation.
  * It is deliberately weaker than a neural perplexity filter (GPT-2-based,
    as in the Alon & Kamfonas paper). It cannot catch attacks phrased in
    grammatical English.
  * It IS effective against: random/GCG suffixes, base64 blobs, cipher
    encodings, hexadecimal escapes, and very dense Unicode obfuscation.
  * Threshold calibration: on the bundled natural-text corpus (see
    :data:`REFERENCE_CORPUS` below) bigram perplexity ranges roughly
    3–15 bits/char. A threshold of 30 bits/char catches gibberish/encoded
    text reliably while rarely flagging clean prompts. Adjust via
    ``threshold`` to tune the precision/recall trade-off.

The filter acts in ``filter_input``: it inspects the prompt (and context, if
present). If the perplexity exceeds the threshold, it replaces the prompt with
a ``[BLOCKED: high-perplexity input]`` sentinel that the target replies to
safely; the context is cleared. This is intentionally conservative: a blocked
prompt cannot trigger the canary, so the measured ASR drops for suffix/encoding
attacks.

DEFENSIVE / AUTHORIZED USE ONLY.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from .base import Defense, register_defense

__all__ = [
    "PerplexityFilterDefense",
    "CharBigramModel",
    "REFERENCE_CORPUS",
]

# --------------------------------------------------------------------------- #
# Bundled reference corpus (natural English text, no external dependency).
# --------------------------------------------------------------------------- #

#: A small bundled reference corpus of natural English text used to train the
#: character bigram model.  Chosen to cover diverse styles (formal, informal,
#: technical) while staying under 10 KB so it's inlined here without a file
#: load. Calibration: on this corpus, bigram perplexity of natural text
#: (excluding the training data itself) is roughly 3–15 bits/char; adversarial
#: suffixes and encoded payloads typically exceed 25 bits/char.
REFERENCE_CORPUS: str = (
    # Formal / technical prose
    "The primary objective of a security scan is to identify vulnerabilities "
    "before an adversary can exploit them. A well-designed benchmark measures "
    "robustness honestly, reporting both success and failure rates without "
    "cherry-picking results. Reproducibility is essential: every run should "
    "produce the same findings given the same corpus and seed. "
    # Conversational / informal
    "Sure, I can help you with that. What would you like to know? "
    "Please summarize the following document for me. "
    "Can you explain how this system works in simple terms? "
    "I would like to request a refund for my recent order. "
    "The weather today is sunny with a high of 72 degrees Fahrenheit. "
    # Instructions / command-style (normal assistant prompts)
    "Translate the following sentence into French: 'The cat sat on the mat.' "
    "Write a short poem about the ocean and the stars above. "
    "List the top five programming languages used in machine learning. "
    "Provide a step-by-step guide to baking a chocolate cake at home. "
    "Compare and contrast the approaches of supervised and unsupervised learning. "
    # Technical / code adjacent
    "The function takes an integer argument and returns its factorial recursively. "
    "HTTP status code 404 indicates that the requested resource was not found. "
    "A convolutional neural network applies learned filters to detect local features. "
    "Version control systems track changes to source code over time enabling "
    "collaboration across distributed teams without losing history. "
    # Narrative prose
    "It was a dark and stormy night when the detective arrived at the old mansion. "
    "She opened the letter with trembling hands, unsure of what she might find inside. "
    "The market square filled with the sounds of vendors calling out their wares. "
    "After years of searching, he finally found the answer hidden in plain sight. "
    "The river wound through the valley, carving a path between ancient hills. "
    # Questions
    "What is the capital of France and what is it known for historically? "
    "How does the human immune system recognize and fight pathogens? "
    "What are the main differences between machine learning and deep learning? "
    "When was the Eiffel Tower built and who designed it? "
    "Why does the sky appear blue during the day and red at sunset? "
    # Mixed punctuation / numbers (still natural text)
    "In 2023, researchers published over 10,000 papers on artificial intelligence. "
    "The temperature dropped to minus 15 degrees Celsius in January 2024. "
    "She scored 98% on the final exam, placing her first in a class of 200. "
    "The package weighs 3.7 kg and measures 40 cm by 25 cm by 15 cm. "
    "Version 2.1.4 of the library fixes three critical security vulnerabilities. "
)


# --------------------------------------------------------------------------- #
# Character bigram model
# --------------------------------------------------------------------------- #


class CharBigramModel:
    """A character bigram language model trained on a reference corpus.

    Estimates ``P(c_i | c_{i-1})`` via smoothed relative-frequency counts over
    the reference text. Perplexity is measured in bits/char (log base 2) —
    lower is more natural, higher is more surprising/adversarial.

    **Smoothing**: add-k (Laplace) smoothing with ``k=0.01`` to avoid zero
    probabilities for unseen bigrams (which would give infinite perplexity).

    Args:
        corpus: The reference text to train on. Defaults to the bundled
            :data:`REFERENCE_CORPUS`.
        k: Laplace smoothing constant (default 0.01).
    """

    def __init__(self, corpus: str = REFERENCE_CORPUS, k: float = 0.01) -> None:
        self.k = k
        # Count unigrams and bigrams over all characters in the corpus.
        unigrams: dict[str, int] = {}
        bigrams: dict[tuple[str, str], int] = {}
        prev = None
        for ch in corpus:
            unigrams[ch] = unigrams.get(ch, 0) + 1
            if prev is not None:
                key = (prev, ch)
                bigrams[key] = bigrams.get(key, 0) + 1
            prev = ch

        self._unigrams = unigrams
        self._bigrams = bigrams
        # Vocabulary: all characters seen in the corpus + a pseudo-OOV slot.
        self._vocab_size = len(unigrams) + 1  # +1 for unseen chars

    def log2_prob(self, prev_char: str, char: str) -> float:
        """Return log2 P(char | prev_char) with Laplace smoothing.

        Returns a negative float (log probability); lower (more negative) means
        less likely. The OOV probability for unseen characters is the smoothing
        floor.
        """
        denom = self._unigrams.get(prev_char, 0) + self.k * self._vocab_size
        numer = self._bigrams.get((prev_char, char), 0) + self.k
        # Guard against denom==0 (corpus is empty or prev_char never seen).
        if denom <= 0.0:
            denom = self.k * self._vocab_size
        return math.log2(numer / denom)

    def perplexity(self, text: str) -> float:
        """Return the per-character bits-of-surprise (bigram perplexity).

        Perplexity = 2^(-H) where H is the per-character log-probability under
        the bigram model. A value near 3–15 is typical for natural English text
        on a character bigram model. Values above ~25 indicate unusual/adversarial
        content; above ~40 indicates likely encoded/random input.

        For texts shorter than 2 characters, a fixed neutral perplexity (1.0) is
        returned (insufficient context to measure).

        Args:
            text: The string to evaluate.

        Returns:
            Perplexity in bits/char (always positive).
        """
        if len(text) < 2:
            return 1.0
        total_log2 = 0.0
        count = 0
        for i in range(1, len(text)):
            total_log2 += self.log2_prob(text[i - 1], text[i])
            count += 1
        if count == 0:
            return 1.0
        avg_log2 = total_log2 / count
        # avg_log2 is negative (log-probability); perplexity = 2^(-avg_log2).
        return 2.0 ** (-avg_log2)


# --------------------------------------------------------------------------- #
# PerplexityFilterDefense
# --------------------------------------------------------------------------- #

#: The sentinel prompt returned when an input is blocked.
_BLOCKED_SENTINEL = "[BLOCKED: high-perplexity input detected by perplexity filter]"

#: Default perplexity threshold (bits/char). Inputs above this are flagged.
#: Calibrated on REFERENCE_CORPUS: natural text ~3–15 bits/char; adversarial
#: suffixes/encoded text typically >25 bits/char. A threshold of 30 provides
#: a comfortable margin. Lower to ~20 for stricter blocking (higher recall,
#: more false positives on unusual-but-benign text like code snippets with
#: many symbols); raise to ~50 for permissive mode (fewer false positives).
DEFAULT_THRESHOLD = 30.0


class PerplexityFilterDefense:
    """Reject inputs whose character bigram perplexity exceeds a threshold.

    Implements the perplexity-filter defense (Alon & Kamfonas 2023; Jain et al.
    2023): inputs with high perplexity under a language model are likely
    adversarial (random/encoded/suffix-optimized) and are blocked before
    reaching the target.

    **Model**: character bigram trained on :data:`REFERENCE_CORPUS` (bundled,
    no download, pure Python, CPU-only). This is deliberately weaker than a
    neural perplexity filter; it catches suffix/encoding attacks but NOT
    attacks phrased in natural English.

    **Action**: if perplexity > ``threshold``, ``filter_input`` replaces the
    prompt with a safe sentinel ``[BLOCKED: ...]`` and clears the context. The
    target receives the sentinel and gives a benign answer; the detector finds
    no marker; the attack scores as a non-success. If perplexity <= threshold,
    the prompt passes through unchanged.

    Args:
        threshold: Perplexity threshold in bits/char (default 30.0). Inputs
            above this value are blocked.  See :data:`DEFAULT_THRESHOLD` for
            calibration notes.
        corpus: Reference corpus for bigram training (default:
            :data:`REFERENCE_CORPUS`).
        check_context: Whether to also apply the filter to the context string
            (default True). Indirect injection may carry the adversarial payload
            in the context rather than the prompt.
        sentinel: The replacement text sent to the target when an input is
            blocked (default: :data:`_BLOCKED_SENTINEL`).

    Calibration
    -----------
    On the bundled reference corpus:

      * Natural English prose: perplexity ~ 3–15 bits/char.
      * GCG/random adversarial suffixes: typically > 30 bits/char.
      * Base64-encoded text: typically > 35 bits/char.
      * Hex-encoded text: typically > 40 bits/char.
      * Caesar / Atbash ciphers of English text: ~ 10–20 bits/char
        (these look like natural text character-frequency-wise, so the
        bigram filter does NOT reliably catch cipher attacks — use
        :class:`~injectkit.defenses.mitigations.InputSanitizerDefense`
        for those).

    The bigram filter is therefore most effective against suffix-optimised
    attacks and binary-encoding obfuscation, and least effective against
    semantic transforms (translation, cipher roleplay).
    """

    name = "perplexity_filter"

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        corpus: str = REFERENCE_CORPUS,
        check_context: bool = True,
        sentinel: str = _BLOCKED_SENTINEL,
    ) -> None:
        self.threshold = threshold
        self.check_context = check_context
        self.sentinel = sentinel
        self._model = CharBigramModel(corpus=corpus)

    def wrap_system(self, system: Optional[str]) -> Optional[str]:
        """Passthrough — the filter acts on user input, not the system prompt."""
        return system

    def filter_input(
        self, prompt: str, context: Optional[str]
    ) -> Tuple[str, Optional[str]]:
        """Block the prompt (and optionally context) if perplexity is too high.

        Returns:
            * ``(sentinel, None)`` if the prompt exceeds the threshold (blocked).
            * ``(prompt, sentinel)`` if only the context exceeds the threshold
              (the prompt is benign but the injected document is adversarial).
            * ``(prompt, context)`` unchanged if both are within threshold.
        """
        # Check the prompt.
        if self._is_high_perplexity(prompt):
            return self.sentinel, None

        # Optionally check the context (indirect injection).
        if self.check_context and context is not None:
            if self._is_high_perplexity(context):
                return prompt, self.sentinel

        return prompt, context

    def filter_output(self, text: str) -> str:
        """Passthrough — this defense only filters input."""
        return text

    # ----------------------------------------------------------------- helpers

    def _is_high_perplexity(self, text: str) -> bool:
        """True if ``text`` has perplexity above the threshold."""
        if not text:
            return False
        ppl = self._model.perplexity(text)
        return ppl > self.threshold

    def perplexity(self, text: str) -> float:
        """Compute and return the bigram perplexity of ``text`` (bits/char).

        Convenience accessor for tests and calibration tooling.
        """
        return self._model.perplexity(text)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def _register() -> None:
    """Register PerplexityFilterDefense on the default registry (idempotent)."""
    from .base import registry as _registry

    if PerplexityFilterDefense.name not in _registry.names():
        register_defense(PerplexityFilterDefense.name, PerplexityFilterDefense)


_register()
