#!/usr/bin/env python

# External imports
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import random

# Local import
from runner import MetricRunner, SentenceContext, read_sentences, mean_confidence_interval

# One-time download of the sentence tokenizer model
#nltk.download("punkt")
#nltk.download("punkt_tab")


class SemanticDiversity(MetricRunner):
    name = "sem"
    title = "Semantic diversity"
    unit = "sentences"
    samples_by_file = [50]
    samples_sweep = [50, 100, 500, 1000, 2000]

    model_name = 'sentence-transformers/all-mpnet-base-v2'
    device = "mps"
    _model = None

    @property
    def model(self):
        # Load the sentence transformer model on first use, then keep it
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def prepare(self, path):
        return SentenceContext(read_sentences(path))

    def capacity(self, ctx):
        return len(ctx.sentences)

    def _embed(self, sentences):
        # Unit-length embeddings, one row per sentence
        emb = self.model.encode(sentences, batch_size=8, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True, device=self.device)
        return list(emb)

    def compute(self, ctx, n):
        x = np.stack(ctx.sample(n, self._embed)).astype(np.float32)   # each sentence embedded once per file

        # Cosine similarity matrix (embeddings are normalized, so this is a dot product)
        similarity = x @ x.T

        # Off-diagonal elements, rescaled from [-1, 1] to a diversity score in [0, 1]
        non_diag = similarity[~np.eye(similarity.shape[0], dtype=bool)]
        return mean_confidence_interval((1 - non_diag) / 2)

    def finish_file(self, ctx):
        torch.mps.empty_cache()


if __name__ == "__main__":
    # usage: python semantic.py [--mode file|sample] [--plot-only] [--import FILE ...]
    random.seed(42)
    SemanticDiversity().main()