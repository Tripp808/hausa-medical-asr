"""
decoding.py

Core decoding functions: Greedy, Shallow Fusion, and Density Ratio Approach.
Used both for evaluation and for the HuggingFace inference pipeline.
"""

import numpy as np
import kenlm
from pyctcdecode import build_ctcdecoder

LN10 = 2.302585  # convert log10 (KenLM) to natural log


def build_decoders(vocab_list, target_lm_path, source_lm_path=None):
    """
    Build the Shallow Fusion decoder and load KenLM models for DRA rescoring.

    Args:
        vocab_list: sorted vocabulary list from the ASR processor
        target_lm_path: path to medical (target domain) KenLM .bin
        source_lm_path: path to general (source domain) KenLM .bin, optional

    Returns:
        dict with 'decoder_sf', 'kenlm_tgt', 'kenlm_src' (None if unavailable)
    """
    decoder_sf = build_ctcdecoder(
        labels=vocab_list,
        kenlm_model_path=target_lm_path,
        alpha=0.8,
        beta=1.5,
    )
    kenlm_tgt = kenlm.Model(target_lm_path)
    kenlm_src = kenlm.Model(source_lm_path) if source_lm_path else None

    return {
        "decoder_sf": decoder_sf,
        "kenlm_tgt": kenlm_tgt,
        "kenlm_src": kenlm_src,
    }


def transcribe_greedy(logits: np.ndarray, processor) -> str:
    """Greedy CTC decoding — no language model."""
    pred_ids = np.argmax(logits, axis=-1)
    return processor.decode(pred_ids)


def transcribe_shallow_fusion(
    logits: np.ndarray,
    decoder_sf,
    lm_weight: float = 0.8,
    word_score: float = 1.5,
    beam_width: int = 100,
) -> str:
    """
    Shallow Fusion decoding:
        log P_SF(W|X) = log P_CTC(W|X) + lm_weight * log P_TGT(W) + word_score * |W|

    Default hyperparameters (lm_weight=0.8, word_score=1.5) are the values
    found via grid search on the evaluation set (SF WER: 23.30%).
    """
    return decoder_sf.decode(
        logits,
        beam_width=beam_width,
        beam_prune_logp=-10.0,
        token_min_logp=-5.0,
    )


def transcribe_density_ratio(
    logits: np.ndarray,
    decoder_sf,
    kenlm_tgt,
    kenlm_src=None,
    lambda_tau: float = 1.2,
    lambda_psi: float = 0.1,
    word_score: float = 0.5,
    beam_width: int = 100,
    nbest: int = 50,
) -> str:
    """
    Density Ratio Approach (n-best rescoring variant):
        log P_DR(W|X) = log P_CTC(W|X)
                      + lambda_tau * log P_TGT(W)
                      - lambda_psi * log P_SRC(W)
                      + word_score * |W|

    Default hyperparameters (lambda_tau=1.2, lambda_psi=0.1, word_score=0.5)
    are the values found via grid search on the evaluation set, reported
    in the paper (DRA WER: 22.93%, vs. Greedy 29.56%, vs. SF 23.30%).
    """
    beams = decoder_sf.decode_beams(
        logits,
        beam_width=beam_width,
        beam_prune_logp=-10.0,
        token_min_logp=-5.0,
    )[:nbest]

    if not beams:
        return ""

    best_text, best_score = beams[0][0], -1e18

    for beam in beams:
        hyp_text = beam[0]
        ctc_logprob = beam[3]

        lp_tgt = kenlm_tgt.score(hyp_text, bos=True, eos=True) * LN10
        lp_src = (kenlm_src.score(hyp_text, bos=True, eos=True) * LN10
                  if kenlm_src else 0.0)

        n_words = max(len(hyp_text.split()), 1)
        score = (ctc_logprob
                 + lambda_tau * lp_tgt
                 - lambda_psi * lp_src
                 + word_score * n_words)

        if score > best_score:
            best_score, best_text = score, hyp_text

    return best_text
