"""
Shared machinery for the diversity-metric scripts (lexical.py, syntactic.py, semantic.py).

A metric is a subclass of MetricRunner that only says how to score one file:

    prepare(path)      read and preprocess one story file -> ctx
    capacity(ctx)      how many units (tokens, sentences, ...) the file offers
    compute(ctx, n)    one random run at sample size n -> (score, ci or None)
    finish_file(ctx)   optional cleanup after each file (e.g. free GPU memory)

Everything else lives here: finding the story directories, skipping results that are
already in the persistent store, plotting, importing old results, and the command line.
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import scipy.stats as stats
from nltk.tokenize import sent_tokenize

from plot import plot_by_section, plot_by_sample, SECTION_SEP
from store import ResultsStore, file_sha1


# ---------------------------------------------------------------------------
# Helpers shared by the metrics
# ---------------------------------------------------------------------------

def mean_confidence_interval(data, confidence=0.95):
    """Mean and the half-width of its confidence interval."""
    data = np.array(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    margin_of_error = sem * stats.norm.ppf((1 + confidence) / 2.)
    return mean, margin_of_error


def read_sentences(path):
    """All sentences in a text file, via the nltk sentence tokenizer."""
    with open(path) as f:
        lines = f.readlines()
    sentences = []
    for t in lines:
        sentences += sent_tokenize(t)
    return sentences


class SentenceContext:
    """
    A file's sentences plus a cache of per-sentence features (parse graphs, embeddings, ...),
    so each sentence is processed at most once per file however many runs sample it.
    """
    def __init__(self, sentences):
        self.sentences = sentences
        self.cache = {}

    def sample(self, n, build):
        """
        Pick n random sentences and return their features in sampled order.
        build(list_of_sentences) -> list of features, called only for sentences not cached yet.
        """
        idx = random.sample(range(len(self.sentences)), n)
        missing = [k for k in idx if k not in self.cache]
        if missing:
            for k, item in zip(missing, build([self.sentences[k] for k in missing])):
                self.cache[k] = item
        return [self.cache[k] for k in idx]


# ---------------------------------------------------------------------------
# Finding the story files
# ---------------------------------------------------------------------------

def list_subdirs(path):
    """Subdirectories of path, skipping any that start with '_' (e.g. _batch, _below_minimum)."""
    return sorted(d for d in os.listdir(path)
                  if os.path.isdir(os.path.join(path, d)) and not d.startswith("_"))


def discover_sections(root, human=True, aigen=True, models=None, group_human=True):
    """
    Returns a list of (section_label, directory).
      human: story/human/<style>              -> "human | <style>"  (or just "<style>")
      aigen: story/aigen/<model>/<style>      -> "<model> | <style>"
    models: optional list of model directory names to include (default: all)
    """
    sections = []
    if human:
        hdir = os.path.join(root, "human")
        for style in list_subdirs(hdir):
            label = f"human{SECTION_SEP}{style}" if group_human else style
            sections.append((label, os.path.join(hdir, style)))
    if aigen:
        adir = os.path.join(root, "aigen")
        for model in list_subdirs(adir):
            if models and model not in models:
                continue
            for style in list_subdirs(os.path.join(adir, model)):
                sections.append((f"{model}{SECTION_SEP}{style}", os.path.join(adir, model, style)))
    return sections


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

class MetricRunner:
    # ----- set these in each metric -----
    name = None                          # short name: results go in <results_root>/<name>/, files are <name>_<time>.txt
    title = None                         # e.g. "Lexical diversity"
    unit = "sentences"                   # what n_samples counts, for plot labels
    samples_by_file = [50]               # sample size(s) for the all-sections plot
    samples_sweep = [50, 100, 500, 1000, 2000]   # sample sizes for the human-only progression plot
    runs = 10

    # ----- shared defaults (override per metric if needed) -----
    story_root = "../story"
    results_root = "../results"
    exclude_prefix = "break_"            # files starting with this are never scored
    interval = "ci"                      # error bars across files in the section plot: "ci" or "sd"
    show_files = False                   # faint dot per story in the section plot

    def __init__(self, story_root=None, results_root=None):
        if self.name is None or self.title is None:
            raise TypeError("MetricRunner subclasses must set name and title")
        self.story_root = story_root or self.story_root
        self.results_root = results_root or self.results_root
        self.results_dir = os.path.join(self.results_root, self.name)
        self.store = ResultsStore(os.path.join(self.results_dir, "data"), sep=SECTION_SEP)

    # ----- override these -----
    def prepare(self, path):
        raise NotImplementedError

    def capacity(self, ctx):
        raise NotImplementedError

    def compute(self, ctx, n):
        raise NotImplementedError

    def finish_file(self, ctx):
        pass

    # ----- files -----
    def discover_sections(self, **kwargs):
        return discover_sections(self.story_root, **kwargs)

    def story_files(self, directory):
        """Story files to score: skips hidden files (.DS_Store) and the excluded prefix."""
        return sorted(f for f in os.listdir(directory)
                      if os.path.isfile(os.path.join(directory, f))
                      and not f.startswith(".") and not f.startswith(self.exclude_prefix))

    # ----- computing -----
    def update_store(self, n_samples, n_runs, sections):
        """Score whatever the store doesn't have yet. Returns {section: [files on disk]}."""
        files_by_section = {}
        computed = cached = 0

        for section, directory in sections:
            files = self.story_files(directory)
            files_by_section[section] = files

            for fname in files:
                path = os.path.join(directory, fname)
                sha1 = file_sha1(path)
                todo = self.store.todo(section, fname, sha1, n_samples, n_runs)
                if not todo:
                    cached += 1
                    continue

                ctx = self.prepare(path)
                available = self.capacity(ctx)
                print(f"{section} / {fname}: {available} {self.unit}, {len(todo)} result(s) to compute")
                too_short = set()

                for n, i in todo:
                    if available < n:
                        if n not in too_short:
                            too_short.add(n)
                            print(f"  skipping n={n}: only {available} {self.unit}")
                        continue
                    score, ci = self.compute(ctx, n)
                    self.store.add(section, fname, sha1, n, i, score, ci)   # on disk immediately
                    computed += 1

                self.finish_file(ctx)

        print(f"Computed {computed} new result(s); {cached} file(s) were already up to date")
        return files_by_section

    # ----- plotting -----
    def plot(self, files_by_section, n_samples, output_file, by_file, show_plot):
        """Plot everything the store holds for the files currently on disk."""
        df = self.store.load_frame(files_by_section, n_samples)
        if df.empty:
            print("No stored results to plot.")
            return

        # Snapshot of exactly what was plotted, readable by plot.py's replot
        snap = df.rename(columns={"filename": "file", "sample_size": "n_samples", "metric": "score"})
        snap[["section", "file", "n_samples", "n_run", "score", "ci"]].to_csv(output_file, index=False)

        png = output_file.replace(".txt", ".png")
        if by_file:
            summary = plot_by_section(
                df,
                title      = f"{self.title} per style",
                xlabel     = "Style",
                ylabel     = f"{self.title} score",
                output     = png,
                interval   = self.interval,
                show_files = self.show_files,
                show       = show_plot,
            )
            summary.to_csv(output_file.replace(".txt", "_summary.csv"), index=False)
        else:
            df["section"] = df["section"].map(lambda s: s.split(SECTION_SEP)[-1])   # "human | scifi" -> "scifi"
            plot_by_sample(
                df,
                title  = f"{self.title} by sample size",
                xlabel = f"Number of {self.unit} sampled (log scale)",
                ylabel = f"{self.title} score",
                output = png,
                show   = show_plot,
            )
        print(f"Results written to {output_file}")

    # ----- importing older output files -----
    def import_legacy(self, path):
        """
        Add rows from an old output file (section,file,n_samples,n_run,score[,ci]) to the store.
        Columns are read by POSITION because older scripts wrote a header without 'section'.
        Rows are matched to the story file on disk (to hash it); rows whose file is gone or
        that are already stored are skipped. Assumes the files haven't changed since scoring.
        """
        dirs, hashes = dict(self.discover_sections()), {}
        added = dup = missing = 0
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader)                                   # header (may be misaligned)
            for row in reader:
                if len(row) < 5:
                    print(f"  skipping row with {len(row)} columns: {row}")
                    continue
                if row[1].isdigit():                               # oldest layout: file,n_samples,n_run,score,ci
                    print(f"  skipping row with no section column: {row}")
                    continue
                section, fname, n, run, score = row[:5]
                ci = row[5] if len(row) > 5 else ""
                key = (section, fname)
                if key not in hashes:
                    full = os.path.join(dirs.get(section, ""), fname)
                    hashes[key] = file_sha1(full) if section in dirs and os.path.isfile(full) else None
                if hashes[key] is None:
                    missing += 1
                    continue
                ok = self.store.add(section, fname, hashes[key], int(n), int(run), float(score),
                                    float(ci) if ci not in ("", "None") else None, skip_existing=True)
                added, dup = added + ok, dup + (not ok)
        print(f"{path}: imported {added}, already stored {dup}, file not on disk {missing}")

    # ----- command line -----
    def main(self, argv=None):
        ap = argparse.ArgumentParser(description=f"{self.title}: score story files and plot the results")
        ap.add_argument("--mode", choices=["file", "sample"], default="file",
                        help="file: every section at one sample size (default); "
                             "sample: human corpora across several sample sizes")
        ap.add_argument("--samples", type=int, nargs="+", help="override the sample size(s)")
        ap.add_argument("--runs", type=int, help=f"random runs per file and size (default {self.runs})")
        ap.add_argument("--models", nargs="+", help="only these model directories (file mode)")
        ap.add_argument("--plot-only", action="store_true", help="plot what is stored; compute nothing")
        ap.add_argument("--show", action="store_true", help="open the plot window")
        ap.add_argument("--import", dest="import_files", nargs="+", metavar="FILE",
                        help="add results from older output files to the store, then exit")
        args = ap.parse_args(argv)

        if args.import_files:
            for p in args.import_files:
                self.import_legacy(p)
            return

        by_file = args.mode == "file"
        if by_file:
            n_samples = args.samples or self.samples_by_file
            sections = self.discover_sections(models=args.models)
        else:
            n_samples = args.samples or self.samples_sweep
            sections = self.discover_sections(aigen=False)   # keeps "human | <style>" labels so the store is shared
        n_runs = args.runs or self.runs

        if args.plot_only:
            files_by_section = {s: self.story_files(d) for s, d in sections}
        else:
            files_by_section = self.update_store(n_samples, n_runs, sections)

        os.makedirs(self.results_dir, exist_ok=True)
        output_file = os.path.join(self.results_dir, f"_{self.name}_{time.strftime('%Y%m%d-%H%M%S')}.txt")
        self.plot(files_by_section, n_samples, output_file, by_file, args.show)