"""
Persistent results store shared by the metric scripts.

One CSV per *series* (the part of a section label before " | ", e.g. "human" or
"sonnet5-5"; a plain label like "scifi" is its own series). Every computed result is
appended to the file immediately, so an interrupted run loses nothing and the next
run only computes what is missing.

Columns: section, file, n_samples, n_run, score, ci, sha1
  sha1 = hash of the story file's contents when the score was computed. If a file
  is edited, its stored rows no longer match and it is recomputed.

The files are valid input for plot.py's replot (the extra sha1 column is ignored).
"""
import csv
import hashlib
import re
from pathlib import Path

import pandas as pd

COLUMNS = ["section", "file", "n_samples", "n_run", "score", "ci", "sha1"]


def file_sha1(path, length=12):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()[:length]


class ResultsStore:
    def __init__(self, data_dir, sep=" | "):
        self.dir = Path(data_dir)
        self.sep = sep
        self._index = {}          # series -> {(section, file): {"sha1": str, "done": {(n, run), ...}}}

    # ----- naming -----
    def series_of(self, section):
        return section.split(self.sep, 1)[0] if self.sep in section else section

    def path(self, section):
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", self.series_of(section)).strip("_") or "series"
        return self.dir / f"{slug}.csv"

    # ----- index of what is already stored -----
    def _load_index(self, section):
        series, path = self.series_of(section), self.path(section)
        if series not in self._index:
            idx = {}
            if path.exists() and path.stat().st_size > 0:
                df = pd.read_csv(path, dtype={"sha1": str})
                for r in df.itertuples(index=False):
                    e = idx.setdefault((r.section, r.file), {"sha1": r.sha1, "done": set()})
                    e["done"].add((int(r.n_samples), int(r.n_run)))
            self._index[series] = idx
        return self._index[series]

    def todo(self, section, fname, sha1, n_samples, n_runs):
        """The (n_samples, n_run) pairs still to compute for this file."""
        idx = self._load_index(section)
        entry = idx.get((section, fname))
        if entry and entry["sha1"] != sha1:
            print(f"  {fname}: file changed since it was last scored, recomputing")
            self.drop_file(section, fname)
            entry = None
        done = entry["done"] if entry else set()
        return [(n, i) for n in n_samples for i in range(n_runs) if (n, i) not in done]

    def add(self, section, fname, sha1, n, run, score, ci=None, skip_existing=False):
        """Append one result to the series file. Returns False if skipped as a duplicate."""
        idx = self._load_index(section)
        entry = idx.get((section, fname))
        if skip_existing and entry and ((n, run) in entry["done"] or entry["sha1"] != sha1):
            return False
        path = self.path(section)
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(COLUMNS)
            w.writerow([section, fname, n, run, round(float(score), 4),
                        "" if ci is None else round(float(ci), 5), sha1])
        idx.setdefault((section, fname), {"sha1": sha1, "done": set()})["done"].add((n, run))
        return True

    def drop_file(self, section, fname):
        """Remove every stored row for one file (used when the file's contents changed)."""
        path = self.path(section)
        df = pd.read_csv(path, dtype={"sha1": str})
        keep = df[~((df["section"] == section) & (df["file"] == fname))]
        tmp = path.with_suffix(".tmp")
        keep.to_csv(tmp, index=False)
        tmp.replace(path)
        self._load_index(section).pop((section, fname), None)

    # ----- reading back for plotting -----
    def load_frame(self, files_by_section, n_samples=None):
        """
        Stored rows for the given {section: [filenames]} (so deleted files drop out),
        optionally limited to the given sample sizes. Columns match
        MetricsCollector.to_dataframe(): section, filename, sample_size, n_run, metric, ci.
        """
        frames = []
        for path in sorted({self.path(s) for s in files_by_section}):
            if path.exists() and path.stat().st_size > 0:
                frames.append(pd.read_csv(path, dtype={"sha1": str}))
        cols = ["section", "filename", "sample_size", "n_run", "metric", "ci"]
        if not frames:
            return pd.DataFrame(columns=cols)
        df = pd.concat(frames, ignore_index=True)
        valid = {(s, f) for s, files in files_by_section.items() for f in files}
        df = df[[(s, f) in valid for s, f in zip(df["section"], df["file"])]]
        if n_samples is not None:
            df = df[df["n_samples"].isin(list(n_samples))]
        df = df.rename(columns={"file": "filename", "n_samples": "sample_size", "score": "metric"})
        return df[cols].reset_index(drop=True)