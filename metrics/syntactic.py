#!python

# External imports
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel.utils import graph_from_networkx
import networkx as nx
import numpy as np
import stanza
from tqdm import tqdm
import random

# Local import
from runner import MetricRunner, SentenceContext, read_sentences, mean_confidence_interval

# Function to create a dependency graph from a Stanza-processed sentence
def create_graph(doc):
    sent = doc.sentences[0]
    G = nx.Graph()

    for word in sent.to_dict():
        if isinstance(word['id'], tuple):
            continue

        # Add the current word to the graph
        if word['id'] not in G.nodes():
            G.add_node(word['id'])

        G.nodes[word['id']]['label'] = word['upos']

        # Add the head of the word (the word it depends on)
        if word['head'] not in G.nodes():
            G.add_node(word['head'])

        G.add_edge(word['id'], word['head'])

    # Set the root node label
    G.nodes[0]['label'] = 'none'

    return G


class SyntacticDiversity(MetricRunner):
    name = "syn"
    title = "Syntactic diversity"
    unit = "sentences"
    samples_by_file = [50]
    samples_sweep = [50, 100, 500, 1000, 2000]

    _nlp = None

    @property
    def nlp(self):
        # Download and initialize the Stanza pipeline on first use
        #stanza.download('en')
        if self._nlp is None:
            self._nlp = stanza.Pipeline('en', processors='tokenize,pos,mwt,lemma,depparse')
        return self._nlp

    def prepare(self, path):
        return SentenceContext(read_sentences(path))

    def capacity(self, ctx):
        return len(ctx.sentences)

    def _parse(self, sentences):
        return [create_graph(self.nlp(s)) for s in tqdm(sentences, leave=False)]

    def compute(self, ctx, n):
        graphs = ctx.sample(n, self._parse)          # each sentence is parsed once per file

        # Convert NetworkX graphs to Grakel format
        G = list(graph_from_networkx(graphs, node_labels_tag='label'))

        # Weisfeiler-Lehman kernel matrix
        gk = WeisfeilerLehman(n_iter=2, normalize=True, base_graph_kernel=VertexHistogram)
        K = np.asarray(gk.fit_transform(G), dtype=np.float32)

        # Off-diagonal elements, rescaled so that higher = more diverse
        non_diag = K[~np.eye(K.shape[0], dtype=bool)]
        return mean_confidence_interval(1 - non_diag)


if __name__ == "__main__":
    # usage: python syntactic.py [--mode file|sample] [--plot-only] [--import FILE ...]
    random.seed(42)
    SyntacticDiversity().main()