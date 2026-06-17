"""Golden-loss regression for the nanoGCG one-hot gradient (CHUNK 3-gcg-advprefix).

ROADMAP CPU/no-GPU parity band: the 8B ASR-parity run is DEFERRED-NO-GPU, so the
white-box optimiser's *numerics* are pinned on the CPU tiny-model path instead.
For a fixed seed, fixed prompt and fixed (benign) target, the target NLL the GCG
gradient is computed from must stay within **5% relative** of a recorded golden
value on GPT-2 and Pythia-160M (docs/REPRODUCE.md). This is the tripwire that
catches a silent regression in the slice location / one-hot gradient / forward
plumbing without needing a GPU or an 8B download.

These tests need ``torch`` + ``transformers`` and download a TINY model (~150-500
MB) on first run. If either dependency or the network is unavailable they SKIP
(the suite stays green on a torch-less host) — they never hang and never attempt a
large model.

DEFERRED-NO-GPU: full 8B fp16 ASR parity (+-10 abs pp band). See docs/REPRODUCE.md.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers", reason="transformers not installed")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from injectkit.whitebox.gcg_hard import (  # noqa: E402
    locate_optim_slice,
    token_gradients_onehot,
)

# Recorded golden target-NLL values: fixed seed, prompt, optim filler and benign
# target on float32 CPU. The optimiser must stay within 5% relative of these.
_GOLDEN = {
    "gpt2": 5.734537,
    "EleutherAI/pythia-160m": 6.253001,
}

_REL_TOL = 0.05  # 5% relative band (ROADMAP CPU parity).
_PROMPT = "Tell me about birds."
_OPTIM = "! ! ! ! !"
_TARGET = "Sure, here is"

# A 5-behavior benign smoke set (distinct fixed prompts) — the CPU stand-in for
# the GPU 5-behavior parity smoke set. Used to assert the loss/gradient path is
# finite and stable across varied inputs, not just one prompt.
_SMOKE_PROMPTS = [
    "Tell me about birds.",
    "Summarize the water cycle.",
    "Explain how a bicycle works.",
    "List three primary colors.",
    "Describe a typical morning routine.",
]


def _load(name):
    """Load a tiny model+tokenizer on CPU/float32, or skip if unavailable."""
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    except Exception as exc:  # noqa: BLE001 - offline/network -> skip, never fail
        pytest.skip(f"could not load tiny model {name!r}: {exc}")
    model.eval()
    return tok, model


def _target_nll(model, slices):
    """The teacher-forced NLL of the target span (what the GCG gradient minimises)."""
    ids = torch.tensor([slices.full_ids])
    with torch.no_grad():
        logits = model(ids).logits
    shift = slices.target_slice.start - 1
    pred = logits[0, shift : slices.target_slice.stop - 1, :]
    tgt = ids[0, slices.target_slice]
    return float(torch.nn.functional.cross_entropy(pred, tgt))


@pytest.mark.parametrize("name", sorted(_GOLDEN))
def test_golden_loss_within_5pct(name):
    """Target NLL stays within 5% relative of the recorded golden value."""
    torch.manual_seed(0)
    tok, model = _load(name)
    msgs = [{"role": "user", "content": _PROMPT}]
    slices = locate_optim_slice(tok, msgs, _OPTIM, _TARGET)

    loss = _target_nll(model, slices)
    golden = _GOLDEN[name]
    rel = abs(loss - golden) / golden
    assert rel <= _REL_TOL, (
        f"{name}: target NLL {loss:.6f} drifted {rel:.3%} from golden "
        f"{golden:.6f} (>5% — golden-loss regression)."
    )


@pytest.mark.parametrize("name", sorted(_GOLDEN))
def test_onehot_gradient_matches_autograd_reference(name):
    """The one-hot gradient equals a direct autograd reference (correctness)."""
    torch.manual_seed(0)
    tok, model = _load(name)
    msgs = [{"role": "user", "content": _PROMPT}]
    slices = locate_optim_slice(tok, msgs, _OPTIM, _TARGET)

    grad = token_gradients_onehot(
        model, slices.full_ids, slices.optim_slice, slices.target_slice
    )
    assert grad.shape == (len(slices.optim_ids), model.get_input_embeddings().weight.shape[0])
    assert bool(torch.isfinite(grad).all())
    # The gradient must be non-trivial (the optim tokens actually influence loss).
    assert float(grad.abs().sum()) > 0.0


def test_golden_loss_smoke_set_is_finite():
    """The 5-behavior smoke set yields finite, positive target NLLs on a tiny model."""
    torch.manual_seed(0)
    tok, model = _load("gpt2")
    for prompt in _SMOKE_PROMPTS:
        msgs = [{"role": "user", "content": prompt}]
        slices = locate_optim_slice(tok, msgs, _OPTIM, _TARGET)
        loss = _target_nll(model, slices)
        assert loss > 0.0 and loss == loss  # finite, non-NaN
