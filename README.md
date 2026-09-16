# Hausa Medical ASR: Text-Only Domain Adaptation

Text-only domain adaptation for low-resource medical ASR using Shallow Fusion
and the Density Ratio Approach, applied to Hausa.

📄 **Paper**: [arXiv link — TBD]
🤗 **Model**: [huggingface.co/USERNAME/hausa-medical-asr](https://huggingface.co/USERNAME/hausa-medical-asr)
🤗 **Dataset**: [huggingface.co/datasets/USERNAME/hausa-medical-speech](https://huggingface.co/datasets/USERNAME/hausa-medical-speech)

## Overview

This repository implements a text-only domain adaptation framework for Hausa
medical speech recognition. The MMS-1B-FL102 acoustic model is used **as-is,
with no fine-tuning** — domain adaptation happens entirely at decoding time
via two language model fusion strategies:

- **Shallow Fusion (SF)** — 21.2% relative WER reduction (29.56% → 23.30%)
- **Density Ratio Approach (DRA)** — 22.4% relative WER reduction (29.56% → 22.93%)

No audio data collection is required for domain adaptation — only text.
Hyperparameters were tuned by grid search on the 250-utterance evaluation
set; see the paper for a full discussion of this limitation.

## Quick Start

```bash
git clone https://github.com/USERNAME/hausa-medical-asr.git
cd hausa-medical-asr
pip install -r requirements.txt
```

```python
from src.inference import HausaMedicalASR

asr = HausaMedicalASR(
    target_lm_path="models/hausa_health_4gram.bin",
    source_lm_path="models/hausa_general_4gram.bin",
)

# Greedy (baseline, no LM)
text = asr.transcribe("audio.wav", method="greedy")

# Shallow Fusion (tuned: lm_weight=0.8, word_score=1.5)
text = asr.transcribe("audio.wav", method="sf", lm_weight=0.8, word_score=1.5)

# Density Ratio Approach (best performing, recommended)
text = asr.transcribe("audio.wav", method="dra",
                       lambda_tau=1.2, lambda_psi=0.1, word_score=0.5)
```

Language models are hosted on HuggingFace and downloaded automatically — see
[Model Weights](#model-weights) below.

## Repository Structure

```
hausa-medical-asr/
├── notebooks/     Colab notebooks for LM training and evaluation
├── src/           Core decoding, metrics, and inference code
├── data/          25 validated evaluation sentences
├── results/       Full evaluation results (WER, CER, MTER)
├── figures/       Paper figures
└── paper/         Preprint PDF
```

## Model Weights

Trained KenLM binaries (medical + general Hausa LMs) are hosted on HuggingFace:

```python
from huggingface_hub import hf_hub_download

target_lm = hf_hub_download("USERNAME/hausa-medical-asr", "hausa_health_4gram.bin")
source_lm = hf_hub_download("USERNAME/hausa-medical-asr", "hausa_general_4gram.bin")
```

## Evaluation Dataset

250 utterances from 10 speakers across six northern Nigerian states, validated
by two medical doctors. Hosted on HuggingFace Datasets:

```python
from datasets import load_dataset
ds = load_dataset("USERNAME/hausa-medical-speech")
```

## Citation

```bibtex
@article{ankeli2027hausamedasr,
  title={Text-Only Domain Adaptation for Low-Resource Medical ASR: Shallow Fusion and Density Ratio Methods for Hausa},
  author={Ankeli, Oche David and Adelani, Hassan and Egbunike, Theodora and [Author 4] and [Author 5]},
  year={2027}
}
```

## License

Code: MIT License. Trained models: CC-BY-4.0. Dataset: see dataset card for
consent-based usage terms.

## Acknowledgements

Medical validation by Dr. Mohammed Mubarak Bello and [Second Annotator],
[Hospital Name]. Speaker recruitment supported by YAAN. Research conducted
within the RUZIVO Research Lab at African Leadership University.
