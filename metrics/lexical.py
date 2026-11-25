#!python

# External imports
from nltk import ngrams
from nltk.tokenize import WordPunctTokenizer
import random
import string
from types import SimpleNamespace

# Local import
from runner import MetricRunner, read_sentences

# Initialize the tokenizer and load punctuation
tokenizer = WordPunctTokenizer()
punctuation = set(string.punctuation)


# Function to compute Type-Token Ratio (TTR)
def ttr(tokens):
    return len(set(tokens)) / len(tokens)


class LexicalDiversity(MetricRunner):
    name = "lex"
    title = "Lexical diversity"
    unit = "tokens"
    samples_by_file = [700]
    samples_sweep = [700, 1000, 5000, 10000, 20000, 40000]

    def prepare(self, path):
        tokens, bi_grams, tri_grams = [], [], []
        for sentence in read_sentences(path):
            token = tokenizer.tokenize(sentence)
            token = [t.lower() for t in token if t not in punctuation]
            tokens += token
            bi_grams += ngrams(token, 2)
            tri_grams += ngrams(token, 3)
        return SimpleNamespace(tokens=tokens, bi_grams=bi_grams, tri_grams=tri_grams)

    def capacity(self, ctx):
        return min(len(ctx.tokens), len(ctx.bi_grams), len(ctx.tri_grams))

    def compute(self, ctx, n):
        # Type-token ratio of n random unigrams, bigrams and trigrams, averaged
        unique_1 = ttr(random.sample(ctx.tokens, n))
        unique_2 = ttr(random.sample(ctx.bi_grams, n))
        unique_3 = ttr(random.sample(ctx.tri_grams, n))
        return (unique_1 + unique_2 + unique_3) / 3, None     # no confidence interval for this metric


if __name__ == "__main__":
    # usage: python lexical.py [--mode file|sample] [--plot-only] [--import FILE ...]
    random.seed(42)
    LexicalDiversity().main()