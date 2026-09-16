"""
inference.py

End-to-end inference: audio file in, transcription out.
This is the same code structure used in the HuggingFace model repo's
inference.py — kept in sync so users can run identically from either location.
"""

import subprocess
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from decoding import (
    build_decoders,
    transcribe_greedy,
    transcribe_shallow_fusion,
    transcribe_density_ratio,
)

MODEL_NAME = "facebook/mms-1b-fl102"


class HausaMedicalASR:
    """
    Convenience wrapper bundling the MMS acoustic model with the
    medical (target) and general (source) KenLM language models
    for Shallow Fusion and Density Ratio decoding.
    """

    def __init__(self, target_lm_path, source_lm_path=None, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = Wav2Vec2Processor.from_pretrained(
            MODEL_NAME, target_lang="hau"
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        self.model.load_adapter("hau")
        self.model.to(self.device).eval()

        vocab_dict = self.processor.tokenizer.get_vocab()
        vocab_list = [tok for tok, _ in sorted(vocab_dict.items(), key=lambda x: x[1])]

        lms = build_decoders(vocab_list, target_lm_path, source_lm_path)
        self.decoder_sf = lms["decoder_sf"]
        self.kenlm_tgt = lms["kenlm_tgt"]
        self.kenlm_src = lms["kenlm_src"]

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Convert any audio format to 16kHz mono and load as array."""
        wav_path = "/tmp/_hausa_asr_tmp.wav"
        subprocess.run(
            f"ffmpeg -y -i '{audio_path}' -ar 16000 -ac 1 -acodec pcm_s16le "
            f"'{wav_path}' -loglevel error",
            shell=True, check=True,
        )
        audio, _ = librosa.load(wav_path, sr=16000)
        return audio

    def _get_logits(self, audio: np.ndarray) -> np.ndarray:
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        return logits[0].cpu().numpy()

    def transcribe(self, audio_path: str, method: str = "dra", **kwargs) -> str:
        """
        Transcribe a Hausa medical audio file.

        Args:
            audio_path: path to audio file (any format ffmpeg supports)
            method: one of 'greedy', 'sf', 'dra'
            **kwargs: decoding hyperparameters, e.g. lambda_tau, lambda_psi

        Returns:
            Transcribed text string
        """
        audio = self._load_audio(audio_path)
        logits = self._get_logits(audio)

        if method == "greedy":
            return transcribe_greedy(logits, self.processor)
        elif method == "sf":
            return transcribe_shallow_fusion(logits, self.decoder_sf, **kwargs)
        elif method == "dra":
            return transcribe_density_ratio(
                logits, self.decoder_sf, self.kenlm_tgt, self.kenlm_src, **kwargs
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'greedy', 'sf', or 'dra'.")


if __name__ == "__main__":
    # Example usage
    asr = HausaMedicalASR(
        target_lm_path="models/hausa_health_4gram.bin",
        source_lm_path="models/hausa_general_4gram.bin",
    )
    result = asr.transcribe("example_audio.wav", method="dra")
    print(f"Transcription: {result}")
