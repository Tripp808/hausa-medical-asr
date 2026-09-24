# Hausa Medical ASR: Text-Only Domain Adaptation

Text-only domain adaptation for low-resource medical ASR using Shallow Fusion
and the Density Ratio Approach (DRA), applied to Hausa.

📄 **Paper**: [arXiv link — TBD]
🤗 **Model**: [huggingface.co/OcheAnkeli/hausa-medical-asr](https://huggingface.co/OcheAnkeli/hausa-medical-asr)
🤗 **Dataset**: [huggingface.co/datasets/OcheAnkeli/hausa-medical-speech](https://huggingface.co/datasets/OcheAnkeli/hausa-medical-speech)

## Overview

This repository implements a text-only domain adaptation framework for Hausa
medical speech recognition. The MMS-1B-FL102 acoustic model with its Hausa CTC
adapter is kept fixed throughout evaluation, with no acoustic-model
fine-tuning. Domain adaptation is performed entirely at decoding time using
lightweight 4-gram language models.

We evaluate three decoding strategies:

- **Greedy decoding** — baseline without an external language model
- **Shallow Fusion (SF)** — incorporates a medical-domain language model during beam search
- **Density Ratio Approach (DRA)** — rescoring of the SF n-best hypotheses using both medical and general-domain language models

The medical language model is trained entirely from text, so no additional
medical speech is required for domain adaptation.

## Results

Evaluation is performed on **250 utterances from 10 speakers**, representing
**25 Hausa medical sentence prompts**. The prompts were independently reviewed
by two Hausa-speaking medical doctors for medical accuracy and Hausa naturalness.
Sentence-level 5-fold cross-validation is used so that tuning prompts are
separated from held-out evaluation prompts.

| System | WER (%) | CER (%) | Relative WER Reduction |
|---|---:|---:|---:|
| Greedy | 29.56 | 6.86 | — |
| Shallow Fusion | 23.55 | 6.07 | 20.3% |
| Density Ratio | 23.33 | 6.08 | 21.1% |

Both SF and DRA significantly improve over greedy decoding (**p < 0.001**).
The 0.22-point difference between DRA and SF is not statistically significant
(**p = 0.423**). Therefore, the results support both text-only adaptation
methods rather than establishing a statistically significant advantage for
DRA.

Confidence intervals and paired significance tests are computed using
sentence-cluster bootstrap resampling, with all recordings of a given sentence
prompt retained together.

## Language Models

### Medical Language Model

A **4-gram KenLM** language model with modified Kneser-Ney smoothing is trained
on **19,421 Hausa health sentences** collected from BBC Hausa and Deutsche
Welle Hausa health sections.

The resulting binary language model is approximately **16.76 MB** and training
takes approximately **2.5 seconds on CPU**.

### General Language Model

The DRA source language model contains **75,372 general-domain Hausa
sentences** collected from four public sources:

- Mozilla Common Voice Hausa
- HausaVisualGenome
- NaijaSenti-Twitter Hausa
- ijdutse Hausa Corpus

The general-domain model provides the source language prior used by DRA during
n-best rescoring.

## Decoding

### Shallow Fusion

Shallow Fusion combines the CTC acoustic-model score with the medical language
model score during beam-search decoding.

A beam width of **100** is used. The language-model weight and word score are
selected by grid search using only the tuning prompts within each cross-
validation fold. The selected parameters are then frozen before decoding the
held-out prompts.

### Density Ratio Approach

DRA performs **n-best rescoring over the SF beam** rather than acting as an
independent first-pass decoder.

For each utterance, the top **50 SF hypotheses** are rescored using:

- the fixed MMS CTC score,
- the medical-domain LM score,
- the general-domain LM score, and
- a word-score term.

DRA parameters are selected using only the tuning prompts within each fold and
are frozen before held-out evaluation.

## Evaluation Dataset

The evaluation set contains:

- **250 utterances**
- **10 speakers**
- **25 sentence prompts**
- **5 male and 5 female speakers**
- Speakers aged **20–35**
- Speakers from **six northern Nigerian states**
- **8 native Hausa speakers and 2 fluent second-language speakers**

The 25 medical prompts cover three levels of linguistic/clinical density:

- 8 short prompts
- 12 medium clinical prompts
- 5 long, dense prompts

Two Hausa-speaking medical doctors independently reviewed the evaluation
sentences. **16 prompts were approved verbatim and 9 were corrected** before
recording.

Speaker identities are represented using anonymized identifiers, and informed
consent was obtained.

## Cross-Validation

The evaluation uses **sentence-level 5-fold cross-validation**.

For each fold:

1. 20 sentence prompts are used for parameter tuning.
2. 5 sentence prompts are held out for evaluation.
3. All 10 speakers remain represented in both sets.
4. SF and DRA parameters are selected only from the tuning prompts.
5. The held-out prompts are decoded using the frozen parameters.

This produces **250 out-of-fold predictions** across the complete evaluation
set while preventing the same sentence prompt from appearing in both tuning
and held-out evaluation within a fold.

## Medical Term Error Rate

We also report **Medical Term Error Rate (MTER)** using a fixed **64-term
medical lexicon**.

MTER is recall-oriented and measures whether medical terms appearing in the
reference are recognized in the hypothesis. It is interpreted alongside WER
and CER rather than as a replacement for them.

Final MTER results are:

| System | MTER (%) |
|---|---:|
| Greedy | 31.27 |
| Shallow Fusion | 24.18 |
| Density Ratio | 24.18 |

The reductions in medical-term error provide complementary evidence that
language-model adaptation improves recognition of medical terminology.

## Language Model Ablation

We also evaluate Shallow Fusion using different language-model training data:

| Condition | WER (%) |
|---|---:|
| General-domain LM only | 23.96 |
| Medical LM only | 23.55 |
| Combined general + medical LM | 23.85 |

The medical LM therefore provides a **0.41 percentage-point WER improvement**
over the general-only SF condition. Combining the two corpora does not improve
over the medical-only condition.

## Domain Contrast Analysis

To verify that the medical and general language models capture different
domain distributions, we compare their log-probability scores on:

- **25 unique medical sentence prompts**, and
- **25 held-out general-domain sentences**.

We compute:

```
log P_MED − log P_GEN
```

The mean score is:

- **Medical sentences:** +3.27
- **General sentences:** −18.04

The difference is statistically significant using a Mann-Whitney U test
(**p < 0.001**).

This analysis provides evidence that the two language models encode a measurable
domain contrast relevant to the density-ratio formulation.

## Reproducibility

The evaluation notebook performs:

- MMS-based Hausa CTC inference
- Greedy decoding
- Shallow Fusion decoding
- DRA n-best rescoring
- Fold-specific parameter tuning
- WER and CER computation
- MTER computation
- Bootstrap confidence intervals
- Statistical significance testing
- Domain-contrast analysis
- Language-model ablations

The main evaluation uses:

- **5 folds**
- **Random seed: 42**
- **Beam width: 100**
- **DRA n-best size: 50**
- Fold-specific parameter selection using the tuning prompts only

See the notebook and accompanying files for the exact parameter grids and
evaluation implementation.

## Repository Structure

```text
hausa-medical-asr/
├── notebooks/       Evaluation and language-model notebooks
├── src/             Decoding, metrics, and inference code
├── data/            Evaluation metadata and validated prompts
├── models/          Language-model binaries
├── results/         WER, CER, MTER, and statistical results
├── figures/         Paper figures
└── paper/           Paper source and PDF
```

## Model Weights

Trained KenLM binaries for the medical and general Hausa language models can
be hosted on Hugging Face.

```python
from huggingface_hub import hf_hub_download

target_lm = hf_hub_download(
    "OcheAnkeli/hausa-medical-asr",
    "hausa_health_4gram.bin"
)

source_lm = hf_hub_download(
    "OcheAnkeli/hausa-medical-asr",
    "hausa_general_4gram.bin"
)
```


## Loading the Evaluation Dataset

If the evaluation dataset is released through Hugging Face Datasets:

```python
from datasets import load_dataset

ds = load_dataset("OcheAnkeli/hausa-medical-speech")
```

The released dataset should follow the consent and usage conditions described
in its dataset card.

## Citation

```bibtex
@inproceedings{ankeli2027hausamedasr,
  title={Text-Only Domain Adaptation for Low-Resource Medical ASR:
         Shallow Fusion and Density Ratio Rescoring for Hausa},
  author={Ankeli, Oche David and Adelani, Hassan and Egbunike, Theodora
          and Gottschalk, Joshua and Dossou, Bonaventure},
  year={2026}
}
```

## License

- **Code**: MIT License.
- **Trained models**: CC-BY-4.0, where applicable.
- **Dataset**: see the dataset card for consent-based usage and redistribution
  conditions.

The underlying BBC Hausa and Deutsche Welle Hausa source text used to train
the medical language model is not redistributed as part of this repository.

## Acknowledgements

We thank the Hausa-speaking medical reviewers who validated the evaluation
sentences and the speakers who contributed the recorded evaluation data.

Research was conducted within the RUZIVO Research Lab at African Leadership
University.

## Disclaimer

This system is a research artifact and has not been clinically validated.
It should not be used as an autonomous clinical documentation or decision-
making system.
