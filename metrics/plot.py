"""
plot.py
---------------
Reusable plotting tools for metric results files.

Two usage patterns:

  1. In-place collection (no file re-read):

        from plot_metrics import MetricsCollector

        collector = MetricsCollector(exclude_prefix="break_")

        for dir in directories:
            collector.set_section(dir)
            for fname in files:
                collector.set_file(fname)
                for n in n_samples:
                    for i in range(n_runs):
                        collector.add(sample_size=n, n_run=i, metric=result)

        collector.plot_by_file(title="Lexical diversity per file", output="results/lex.png")
        collector.plot_by_sample(title="Lexical diversity by sample size")

  2. Post-hoc file parsing:

        from plot_metrics import load_file_data, plot_by_file

        df = load_file_data("results/lex_20261120.csv", exclude_prefix="break_")
        plot_by_file(df, title="Lexical diversity per file", output="results/lex.png")
"""

import re
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

import re
import shutil
from pathlib import Path

def duplicate_cleaned(src_path: str) -> Path:
    src = Path(src_path)
    # Remove leading underscores OR '_YYYYMMDD-HHMMSS' anywhere in the stem
    cleaned_stem = re.sub(r'^_+|_\d{8}-\d{6}', '', src.stem)
    dst = str(src.with_name(f"{cleaned_stem}{src.suffix}"))

    shutil.copy2(src, dst)
    return dst

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def set_theme():
    """
    Apply a clean theme close to ggplot2's theme_minimal:
    white background, light grey gridlines, no top/right spines,
    and a neutral sans-serif font.
    """
    sns.set_theme(
        style   = "whitegrid",
        palette = "muted",
        font    = "DejaVu Sans",
        rc      = {
            "axes.facecolor":     "white",
            "figure.facecolor":   "white",
            "axes.edgecolor":     "#222222",
            "axes.linewidth":     1.3,
            "axes.spines.top":    False,
            "axes.spines.bottom": True,
            "axes.spines.left":   True,
            "axes.spines.right":  False,
            "grid.color":         "#DDDDDD",
            "grid.linewidth":     1.0,
            "axes.titlesize":     17,
            "axes.titleweight":   "bold",
            "axes.labelsize":     18,
            "axes.labelweight":   "bold",
            "axes.labelcolor":    "#000000",
            "xtick.labelsize":    16,
            "ytick.labelsize":    12,
            "xtick.color":        "#111111",
            "ytick.color":        "#111111",
            "xtick.major.size":   6,
            "ytick.major.size":   6,
            "xtick.major.width":  1.3,
            "ytick.major.width":  1.3,
            "legend.fontsize":    17,
            "text.color":         "#111111",
            "figure.titlesize":   25,
            "figure.titleweight": "bold",
        }
    )


# ---------------------------------------------------------------------------
# In-place collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Accumulates metric rows during processing so the output file
    does not need to be re-opened for plotting.

    Parameters
    ----------
    exclude_prefix : filenames starting with this string are ignored
    """

    def __init__(self, exclude_prefix: str = None):
        self._rows           = []
        self._section        = None
        self._filename       = None
        self._exclude_prefix = exclude_prefix

    def set_section(self, section: str):
        """Call when entering a new directory / section."""
        self._section = section.split("/")[-1]

    def set_file(self, filename: str):
        """Call when starting a new file."""
        self._filename = filename.split("/")[-1]

    def add(self, sample_size: int, n_run: int, metric: float, ci: float = None):
        """Record one metric value. Skips files matching exclude_prefix."""
        if self._exclude_prefix and self._filename.startswith(self._exclude_prefix):
            return
        self._rows.append({
            "section":     self._section,
            "filename":    self._filename,
            "sample_size": sample_size,
            "n_run":       n_run,
            "metric":      metric,
            "ci":          ci,
        })

    def to_dataframe(self) -> pd.DataFrame:
        """Return all collected rows as a DataFrame."""
        return pd.DataFrame(self._rows)

    def plot_by_file(self, title: str = "Mean metric per file",
                     xlabel: str = None, ylabel: str = "Mean metric",
                     output: str = None, height: float = 5, aspect: float = 0.8, show: bool = True):
        """Plot mean metric per file from collected data."""
        plot_by_file(self.to_dataframe(), title=title, xlabel=xlabel, ylabel=ylabel,
                     output=output, height=height, aspect=aspect, show=show)

    def plot_by_sample(self, title: str = "Mean metric per sample size",
                       xlabel: str = "Sample size", ylabel: str = "Mean metric",
                       output: str = None, figsize: tuple = (7, 5), show: bool = True):
        """Plot mean metric per sample size from collected data."""
        plot_by_sample(self.to_dataframe(), title=title, xlabel=xlabel, ylabel=ylabel,
                       output=output, figsize=figsize, show=show)

    def plot_by_section(self, **kwargs):
        return plot_by_section(self.to_dataframe(), **kwargs)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_data_row(line: str) -> dict | None:
    """
    Parse a single data line in either format:
        sample_size:n_run: metric
        sample_size:n_run: metric +- ci
    Returns a dict or None if the line doesn't match.
    """
    m = re.match(r"(\d+):(\d+):\s([0-9.]+)(?:\s\+-\s([0-9.]+))?", line)
    if not m:
        return None
    return {
        "sample_size": int(m.group(1)),
        "n_run":       int(m.group(2)),
        "metric":      float(m.group(3)),
        "ci":          float(m.group(4)) if m.group(4) else None,
    }


def _load_legacy_sample_data(filepath: str) -> pd.DataFrame:
    """
    Parse a file of lines in the format:
        sample_size:n_run: metric
        sample_size:n_run: metric +- ci   (ci column present if any row has it)

    Returns a DataFrame with columns: sample_size, n_run, metric[, ci].
    """
    rows = []
    with open(filepath) as f:
        for line in f:
            row = _parse_data_row(line.strip())
            if row:
                rows.append(row)
    return pd.DataFrame(rows)


def _load_legacy_file_data(filepath: str, exclude_prefix: str = None) -> pd.DataFrame:
    """
    Parse a file with optional section headers (*** dir ***) and filename
    headers, followed by lines in the format:
        sample_size:n_run: metric
        sample_size:n_run: metric +- ci   (ci column present if any row has it)

    Returns a DataFrame with columns: section, filename, sample_size, n_run, metric[, ci].
    Rows whose filename starts with exclude_prefix are dropped.
    """
    rows = []
    section = None
    filename = None

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Section header: *** path ***
            m_section = re.match(r"\*+\s+(.*?)\s+\*+", line)
            if m_section:
                section = m_section.group(1).split("/")[-1]
                continue

            # Filename header: ends in .txt (no colon)
            if line.endswith(".txt") and ":" not in line:
                filename = line.split("/")[-1]
                continue

            # Data row
            row = _parse_data_row(line)
            if row and filename:
                rows.append({"section": section, "filename": filename, **row})

    df = pd.DataFrame(rows)

    if exclude_prefix:
        df = df[~df["filename"].str.startswith(exclude_prefix)]

    return df


# ---------------------------------------------------------------------------
# CSV results files (written by the metric scripts: section,file,n_samples,n_run,score)
# ---------------------------------------------------------------------------

_CSV_ALIASES = {"file": "filename", "n_samples": "sample_size", "score": "metric"}
_FRAME_COLS  = ["section", "filename", "sample_size", "n_run", "metric", "ci"]


def _is_csv_results(filepath: str) -> bool:
    """True if the first non-blank line is a CSV header with n_run and a score/metric column."""
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cols = [c.strip().lower() for c in line.split(",")]
                return "n_run" in cols and ("score" in cols or "metric" in cols)
    return False


def _load_csv_results(filepath: str, exclude_prefix: str = None) -> pd.DataFrame:
    """
    Read a CSV results file into the same frame MetricsCollector.to_dataframe() produces:
    columns section, filename, sample_size, n_run, metric, ci.

    Accepted header names (case-insensitive):
        section (optional), file | filename, n_samples | sample_size,
        n_run, score | metric, ci (optional)
    """
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={old: new for old, new in _CSV_ALIASES.items()
                            if old in df.columns and new not in df.columns})

    missing = {"filename", "sample_size", "n_run", "metric"} - set(df.columns)
    if missing:
        raise ValueError(f"{filepath}: missing column(s) {sorted(missing)}; found {list(df.columns)}")

    base = lambda s: s.split("/")[-1] if isinstance(s, str) else s    # same as the collector
    df["filename"] = df["filename"].map(base)
    df["section"]  = df["section"].map(base) if "section" in df.columns else None
    if "ci" not in df.columns:
        df["ci"] = None

    df = df[_FRAME_COLS]
    if exclude_prefix:
        df = df[~df["filename"].astype(str).str.startswith(exclude_prefix)]
    return df.reset_index(drop=True)


def load_sample_data(filepath: str, exclude_prefix: str = None) -> pd.DataFrame:
    """
    Load results for plot_by_sample().

    New CSV format: returns the full frame (section, filename, sample_size, n_run,
    metric, ci), so plot_by_sample draws one line per file, as it does when plotting
    in place from a MetricsCollector.

    Legacy text format (lines of "sample_size:n_run: metric"): returns
    sample_size, n_run, metric[, ci] as before.
    """
    if _is_csv_results(filepath):
        return _load_csv_results(filepath, exclude_prefix)
    return _load_legacy_sample_data(filepath)


def load_file_data(filepath: str, exclude_prefix: str = None) -> pd.DataFrame:
    """
    Load results for plot_by_file() / plot_by_section().

    Reads the new CSV format (section,file,n_samples,n_run,score) or, if the file has
    no CSV header, the legacy text format with "*** dir ***" and filename headers.
    Rows whose filename starts with exclude_prefix are dropped.
    """
    if _is_csv_results(filepath):
        return _load_csv_results(filepath, exclude_prefix)
    return _load_legacy_file_data(filepath, exclude_prefix)


def replot(filepath: str, mode: str = "section", title: str = None, ylabel: str = "Mean metric",
           output: str = None, exclude_prefix: str = None, show: bool = True, **kwargs):
    """
    Replot a saved results file without rerunning the metric script.

    mode   : "section" (human files as points, models aggregated by style),
             "file"    (one point per file, faceted by section),
             "sample"  (metric against sample size)
    kwargs : passed to the plot function (e.g. interval="sd", show_files=True)
    """
    if mode == "sample":
        df = load_sample_data(filepath, exclude_prefix)
        return plot_by_sample(df, title=title or "Metric by sample size",
                              xlabel=kwargs.pop("xlabel", "Number of tokens sampled (log scale)"),
                              ylabel=ylabel, output=output, show=show, **kwargs)
    df = load_file_data(filepath, exclude_prefix)
    if mode == "file":
        return plot_by_file(df, title=title or "Mean metric per file",
                            ylabel=ylabel, output=output, show=show, **kwargs)
    if mode == "section":
        return plot_by_section(df, title=title or "Mean metric by style",
                               ylabel=ylabel, output=output, show=show, **kwargs)
    raise ValueError("mode must be 'section', 'file' or 'sample'")

# ---------------------------------------------------------------------------
# Summarisers
# ---------------------------------------------------------------------------

def _has_ci(df: pd.DataFrame) -> bool:
    """True if the dataframe has a ci column with at least some non-null, non-zero values."""
    return "ci" in df.columns and df["ci"].replace(0, None).notna().any()


def _summarise(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Summarise metric by group_cols.
    Uses inverse-variance weighting when ci is present, simple mean otherwise.
    Returns columns: group_cols + [mean_metric] + optionally [wse].
    """
    if _has_ci(df):
        valid = df[df["ci"].replace(0, None).notna()].copy()
        valid["w"] = 1 / valid["ci"] ** 2

        def weighted(g):
            w = g["w"];  m = g["metric"]
            wmean = (m * w).sum() / w.sum()
            wse   = (1 / w.sum()) ** 0.5
            return pd.Series({"mean_metric": wmean, "wse": wse})

        return (
            valid.groupby(group_cols)
            .apply(weighted)
            .reset_index()
            .sort_values(group_cols[-1])
        )
    else:
        def ci_from_runs(g):
            n    = len(g)
            mean = g["metric"].mean()
            se   = g["metric"].std() / (n ** 0.5)
            # t critical value for 95% CI; falls back to 0 if only one run
            t_crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else 0
            return pd.Series({"mean_metric": mean, "wse": t_crit * se})

        return (
            df.groupby(group_cols)
            .apply(ci_from_runs)
            .reset_index()
            .sort_values(group_cols[-1])
        )

def summarise_by_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise metric per sample size, grouped by filename if multiple files present."""
    has_files = "filename" in df.columns and df["filename"].nunique() > 1
    has_sections = "section" in df.columns and df["section"].notna().any()
    if has_files:
        group_cols = ["section", "filename", "sample_size"] if has_sections else ["filename", "sample_size"]
    else:
        group_cols = ["sample_size"]
    return _summarise(df, group_cols)


def summarise_by_file(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise metric per file (and section, if present)."""
    group_cols = ["section", "filename"] if "section" in df.columns and df["section"].notna().any() \
                 else ["filename"]
    return _summarise(df, group_cols)

# Fixed colours for named series, so a model keeps the same colour in every plot and every
# metric. Series not listed here take the next free colour from the palette.
SERIES_COLORS = {
    "sonnet5-5":         sns.color_palette("muted")[2],   # green
    "gemini-4-hydrogen": sns.color_palette("muted")[3],   # red
}

# AI prompts with no human author file -> the human style directory whose group average is
# drawn beside them (open circles) in the "generalized" panel. {} switches it off.
GENRE_GROUPS = {"scifi": "scifi", "non-scifi": "fiction"}

# Panel display names (data keys are untouched) and left-to-right panel order.
PANEL_ORDER  = ["generalized", "fiction", "scifi"]

# NOTE quick and dirty approach, STYLES is copied from ../preprocess/openrouter.py
# Full names for the author clusters in the "scifi" / "non-scifi" panels (keys are the lower-case
# file / style names; anything not listed keeps its file name).
STYLES = {
    # science fiction authors
    "verne": "Jules Verne",
    "kipling" : "Rudyard Kipling",
    "wells" : "H. G. Wells",
    "lovecraft": "H.P. Lovecraft",
    "bradbury": "Ray Bradbury",
    # non-science fiction writers
    "james": "Henry James",
    "doyle": "Arthur C. Doyle",
    "woolf": "Virginia Woolf",
    "joyce": "James Joyce",
    "hemingway": "Ernest Hemingway",

}


def _group_err(vals, interval: str = "ci", level: float = 0.95) -> float:
    """Error-bar half-width across values: t-based CI of the mean, or the SD."""
    vals = pd.Series(list(vals), dtype=float).dropna()
    n = len(vals)
    if n < 2:
        return 0.0
    sd = float(vals.std(ddof=1))
    return sd if interval == "sd" else float(stats.t.ppf(0.5 + level / 2, df=n - 1) * sd / n ** 0.5)


def summarise_group_averages(df: pd.DataFrame, genre_groups: dict = None,
                             individual_groups=("human",),
                             interval: str = "ci", level: float = 0.95) -> pd.DataFrame:
    """
    Group averages for AI styles that have no human file (e.g. "scifi", "non-scifi").

    genre_groups maps such an AI style to a human style directory
    (default GENRE_GROUPS: "non-scifi" -> "fiction", "scifi" -> "scifi"). For each entry:
      * one human row per individual group: the mean over ALL raw rows of that directory's files
      * one row per model: the mean over ALL raw rows of that model's style directories
        whose name matches a human file in that directory (doyle, hemingway, ... for "fiction")

    The averages use the raw rows (every run of every file pooled), not the per-file summaries;
    err is the CI or SD across those raw rows, and n_files counts the files that went in.

    Returns columns: group, style, label, mean_metric, err, n_files, kind ("group"),
    anchor (the AI style the average is drawn beside).
    """
    cols = ["group", "style", "label", "mean_metric", "err", "sd", "n_files", "kind", "anchor"]
    gmap = {k.lower(): v.lower() for k, v in (GENRE_GROUPS if genre_groups is None else genre_groups).items()}
    individual_groups = set(individual_groups or ())
    d = df[df["section"].notna()].copy()
    split = d["section"].map(_split_section)
    d["_g"], d["_style"] = split.map(lambda t: t[0]), split.map(lambda t: t[1].strip().lower())
    d["_file"] = d["filename"].astype(str).str.replace(r"\.txt$", "", regex=True).str.strip().str.lower()

    is_ind   = d["_g"].isin(individual_groups)
    human, other = d[is_ind], d[~is_ind]
    human_labels = set(human["_file"])                 # AI styles matching one of these sit in a human panel
    rows = []
    for ai_key, genre in gmap.items():
        if ai_key in human_labels or ai_key not in set(other["_style"]):
            continue
        hum = human[human["_style"] == genre]
        if hum.empty:
            print(f"NOTE: no human '{genre}' directory, so no group average beside '{ai_key}'")
            continue
        label    = f"avg {genre}"
        hum_keys = set(hum["_file"])
        for g, raw in hum.groupby("_g", dropna=False):                       # human average
            rows.append(dict(group=None if pd.isna(g) else g, style=genre, label=label,
                             mean_metric=raw["metric"].mean(), err=_group_err(raw["metric"], interval, level),
                             sd=_group_err(raw["metric"], "sd"),
                             n_files=raw.groupby(["section", "filename"]).ngroups, kind="group", anchor=ai_key))
        for g, mg in other.groupby("_g", dropna=False):                      # each model's average
            raw = mg[mg["_style"].isin(hum_keys)]
            if len(raw):
                rows.append(dict(group=None if pd.isna(g) else g, style=genre, label=label,
                                 mean_metric=raw["metric"].mean(), err=_group_err(raw["metric"], interval, level),
                                 sd=_group_err(raw["metric"], "sd"),
                                 n_files=raw.groupby(["section", "filename"]).ngroups, kind="group", anchor=ai_key))
    return pd.DataFrame(rows, columns=cols)


def summarise_points(df: pd.DataFrame, individual_groups=("human",),
                     interval: str = "ci", level: float = 0.95,
                     genre_groups: dict = None) -> pd.DataFrame:
    """
    One row per plotted point.
      - sections whose group is in individual_groups -> one point PER FILE
        (err = the file's own run-based CI, as in the old per-file plot)
      - all other sections -> one point per section, aggregated across files
        (err = CI or SD across files, see summarise_by_section)

      - plus group averages for AI styles with no human file (summarise_group_averages)

    Returns columns: group, style, label, mean_metric, err, sd, n_files, kind ("file"/"section"/"group"),
    anchor (group rows only: the AI style the average is drawn beside)
    """
    individual_groups = set(individual_groups or ())
    sections = df["section"].dropna().unique()
    ind = [s for s in sections if _split_section(s)[0] in individual_groups]
    agg = [s for s in sections if s not in ind]
    parts = []

    if ind:
        pf = summarise_by_file(df[df["section"].isin(ind)]).copy()
        pf["group"] = pf["section"].map(lambda s: _split_section(s)[0])
        pf["style"] = pf["section"].map(lambda s: _split_section(s)[1])
        pf["label"] = pf["filename"].str.replace(r"\.txt$", "", regex=True)
        pf = pf.rename(columns={"wse": "err"})
        pf["n_files"] = 1
        pf["sd"]      = float("nan")      # a single file has no between-file SD (its err is the run-based CI)
        pf["kind"]    = "file"
        parts.append(pf[["group", "style", "label", "mean_metric", "err", "sd", "n_files", "kind"]])

    if agg:
        ag = summarise_by_section(df[df["section"].isin(agg)], interval=interval, level=level).copy()
        ag["group"] = ag["section"].map(lambda s: _split_section(s)[0])
        ag["style"] = ag["section"].map(lambda s: _split_section(s)[1])
        ag["label"] = ag["style"]
        ag["kind"]  = "section"
        parts.append(ag[["group", "style", "label", "mean_metric", "err", "sd", "n_files", "kind"]])

    ga = summarise_group_averages(df, genre_groups, individual_groups, interval, level)
    if len(ga):
        parts.append(ga)

    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Plotters
# ---------------------------------------------------------------------------

def plot_by_sample(
    df:       pd.DataFrame,
    title:    str = "Mean metric per sample size",
    xlabel:   str = "Sample size",
    ylabel:   str = "Mean metric",
    output:   str = None,
    figsize:  tuple = (11, 6),
    log_x:    bool = True,
    show:     bool = True
) -> None:
    """
    Scatter plot of mean metric vs sample size (numeric x axis).

    Parameters
    ----------
    df      : raw DataFrame from load_sample_data()
    title   : plot title
    xlabel  : x axis label (default: "Sample size")
    ylabel  : y axis label (default: "Mean metric")
    output  : filepath to save the figure (optional; displays if None)
    figsize : (width, height) in inches
    log_x   : use a log scale on the x axis (default: True)
    """
    set_theme()
    summary = summarise_by_sample(df)
    has_files = "filename" in summary.columns

    # marker shapes cycled per section, colours per file
    marker_shapes = ["o", "s", "^", "D", "v", "P", "X", "*"]
    has_sections  = "section" in summary.columns and summary["section"].notna().any()

    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("muted", n_colors=summary["filename"].nunique() if has_files else 1)

    if has_files:
        sections      = sorted(summary["section"].unique()) if has_sections else [None]
        section_marker = {s: marker_shapes[i % len(marker_shapes)] for i, s in enumerate(sections)}

        for (fname, colour) in zip(summary["filename"].unique(), palette):
            data   = summary[summary["filename"] == fname].sort_values("sample_size")
            section = data["section"].iloc[0] if has_sections else None
            marker  = section_marker.get(section, "o")
            ax.plot(data["sample_size"], data["mean_metric"],
                    color=colour, linewidth=2, zorder=1)
            ax.scatter(data["sample_size"], data["mean_metric"],
                       s=70, color=colour, marker=marker, zorder=2, label=fname)
            if "wse" in data.columns:
                ax.errorbar(
                    data["sample_size"], data["mean_metric"],
                    yerr=data["wse"], fmt="none", color=colour, capsize=4, linewidth=1.6
                )

        # legend: file colours + section shape patches
        handles, labels = ax.get_legend_handles_labels()
        if has_sections:
            import matplotlib.lines as mlines
            for section, marker in section_marker.items():
                handles.append(mlines.Line2D([], [], color="grey", marker=marker,
                                             linestyle="none", markersize=9, label=f"[{section}]"))
        ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    else:
        ax.plot(summary["sample_size"], summary["mean_metric"],
                color=palette[0], linewidth=2, zorder=1)
        ax.scatter(summary["sample_size"], summary["mean_metric"],
                   s=70, color=palette[0], zorder=2)
        if "wse" in summary.columns:
            ax.errorbar(
                summary["sample_size"], summary["mean_metric"],
                yerr=summary["wse"], fmt="none", color=palette[0], capsize=4, linewidth=1.6
            )

    ax.margins(x=0.08)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    sample_sizes = sorted(summary["sample_size"].unique())
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels(
        [f"{n:,}" for n in sample_sizes],
        rotation=45, ha="right"
    )
    # suppress minor ticks that log scale adds by default
    ax.xaxis.set_minor_locator(plt.NullLocator())

    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        duplicate_cleaned(output)

    if show:
        plt.show()


def plot_by_file(
    df:       pd.DataFrame,
    title:    str = "Mean metric per file",
    xlabel:   str = None,
    ylabel:   str = "Mean metric",
    output:   str = None,
    height:   float = 6,
    aspect:   float = 0.9,
    show:     bool = True,
) -> None:
    """
    Scatter plot of mean metric per file, faceted by section if present.

    Parameters
    ----------
    df      : raw DataFrame from load_file_data()
    title   : plot title
    xlabel  : x axis label (default: None, i.e. no label)
    ylabel  : y axis label (default: "Mean metric")
    output  : filepath to save the figure (optional; displays if None)
    height  : height of each facet panel in inches
    aspect  : width/height ratio of each facet panel
    """
    summary = summarise_by_file(df)
    has_sections = "section" in summary.columns and summary["section"].notna().any()

    set_theme()

    if has_sections:
        sections = sorted(summary["section"].unique())
        fig, axes = plt.subplots(
            1, len(sections),
            figsize = (len(sections) * height * aspect, height),
            sharey  = True
        )
        if len(sections) == 1:
            axes = [axes]

        for ax, section in zip(axes, sections):
            data = summary[summary["section"] == section].sort_values("filename")
            x_pos = range(len(data))
            sns.scatterplot(data=data, x="filename", y="mean_metric", s=80, ax=ax)
            if "wse" in data.columns:
                ax.errorbar(
                    x_pos, data["mean_metric"],
                    yerr=data["wse"], fmt="none", color="steelblue", capsize=4, linewidth=1.6
                )
            ax.set_title(section)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel if ax == axes[0] else "")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(data["filename"], rotation=45, ha="right")

        fig.suptitle(title, y=1.02)
        if xlabel:
            fig.supxlabel(xlabel, fontsize=18, fontweight="bold", color="#000000")
        plt.tight_layout()
        # fig.subplots_adjust(wspace=0.05)
        if output:
            fig.savefig(output, dpi=150, bbox_inches="tight")
            duplicate_cleaned(output)
        
        if show:
            plt.show()

    else:
        fig, ax = plt.subplots(figsize=(max(6, len(summary) * 0.6), height))
        x_pos = range(len(summary))
        sns.scatterplot(data=summary, x="filename", y="mean_metric", s=80, ax=ax)
        if "wse" in summary.columns:
            ax.errorbar(
                x_pos, summary["mean_metric"],
                yerr=summary["wse"], fmt="none", color="steelblue", capsize=4, linewidth=1.6
            )
        ax.set_xlabel(xlabel or "")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary["filename"], rotation=45, ha="right")
        plt.tight_layout()
        if output:
            plt.savefig(output, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

# ---------------------------------------------------------------------------
# Section-level aggregation (one point per style directory)
# ---------------------------------------------------------------------------

def summarise_by_section(df: pd.DataFrame, interval: str = "ci", level: float = 0.95) -> pd.DataFrame:
    """
    Two-stage summary:
      1. each file is reduced to a single value (summarise_by_file, as before)
      2. the per-file values are aggregated within each section

    The uncertainty in stage 2 is the spread BETWEEN files (stories), not the
    within-file ci, because the files are the independent samples of a style.

    interval : "ci" -> t-based confidence interval of the mean (half-width)
               "sd" -> sample standard deviation across files
    level    : confidence level when interval="ci" (default 0.95)

    Returns columns: section, mean_metric, err, sd, n_files
    (err is the half-width of the bar to draw; 0 when a section has one file).
    """
    if interval not in ("ci", "sd"):
        raise ValueError("interval must be 'ci' or 'sd'")

    per_file = summarise_by_file(df)
    if "section" not in per_file.columns:
        raise ValueError("No section column found; summarise_by_section needs section data.")

    rows = []
    for section, g in per_file.groupby("section"):
        vals = g["mean_metric"].dropna()
        n    = len(vals)
        sd   = vals.std(ddof=1) if n > 1 else 0.0
        if interval == "sd":
            err = sd
        else:
            err = stats.t.ppf(0.5 + level / 2, df=n - 1) * sd / (n ** 0.5) if n > 1 else 0.0
        rows.append({"section": section, "mean_metric": vals.mean(),
                     "err": err, "sd": sd, "n_files": n})
    return pd.DataFrame(rows)


SECTION_SEP = " | "   # separates group and style in a section label

def _split_section(section: str):
    """'sonnet5-5 | hemingway' -> ('sonnet5-5', 'hemingway'); 'scifi' -> (None, 'scifi')."""
    if SECTION_SEP in section:
        group, style = section.split(SECTION_SEP, 1)
        return group, style
    return None, section


def plot_by_section(
    df:                pd.DataFrame,
    title:             str   = "Mean metric",
    xlabel:            str   = "Style",
    ylabel:            str   = "Mean metric",
    output:            str   = None,
    individual_groups: tuple = ("human",),
    interval:          str   = "ci",
    level:             float = 0.95,
    sort_by:           str   = "name",
    show_files:        bool  = False,
    label_rotation:    float = 45,
    height:            float = 6.5,
    show:              bool  = True,
    series_colors:     dict  = None,
    genre_groups:      dict  = None,
    bar_baseline:      float = None,
    side_panel_scale:  float = 0.7,
) -> pd.DataFrame:
    """
    Clusters by style name, one panel per human style directory.

      * One PANEL per human directory ("scifi", "fiction").
      * Inside a panel, one CLUSTER per human file.
      * Each cluster holds one point per SERIES: the human file itself ("human")
        plus every model's point for the style directory with the same name
        (matched case-insensitively). Series are colour-coded, and every point gets
        its own angled x-axis label (with file count included for AI series).
      * AI styles that match no human file are collected in an extra "generalized" panel.
        There, each style listed in genre_groups (default GENRE_GROUPS) also gets group averages,
        drawn as OPEN circles in the series colour, computed from the RAW rows (all runs of all files pooled;
        error bar = CI or SD across those rows): the human directory's average, and each model's
        average over its styles that have a human file in that directory
        ("non-scifi" -> human "fiction", "scifi" -> human "scifi"). genre_groups={} turns it off.
      * The "generalized" panel is a BAR chart (error bars = standard deviation): filled bars are the
        group averages, hollow bars the scifi / non-scifi style data. Bars start at bar_baseline;
        None (default) starts them at the bottom of the shared y axis, 0 gives true zero-based bars.
      * side_panel_scale (default 0.7) narrows the human panels relative to "generalized"; the
        figure gets narrower by the same amount. 1.0 restores the previous widths.
    """
    set_theme()
    pts = summarise_points(df, individual_groups, interval, level, genre_groups)
    pts["group"]  = pts["group"].fillna("")
    pts["key"]    = pts["label"].str.strip().str.lower()
    is_grp        = pts["kind"] == "group"
    pts.loc[is_grp, "key"] = pts.loc[is_grp, "anchor"].str.strip().str.lower()   # group averages sit beside their AI style
    pts["series"] = pts["group"].replace("", "all")

    ind = pts[pts["kind"] == "file"].copy()
    oth = pts[pts["kind"] == "section"].copy()

    found   = set(ind["group"])
    missing = [g for g in (individual_groups or ()) if g not in found]
    if missing:
        print(f"NOTE: no per-file points for group(s) {missing}. Section labels seen: "
              f"{sorted(df['section'].dropna().unique())[:12]} "
              f"(per-file points need labels like 'human{SECTION_SEP}scifi')")

    # ----- panels: (title, [(label, key), ...], human series name or None) -----
    sort_col  = "mean_metric" if sort_by == "mean" else "label"
    ind_order = {g: i for i, g in enumerate(individual_groups or ())}
    ind["_g"] = ind["group"].map(lambda g: ind_order.get(g, 99))
    ind = ind.sort_values(["_g", "group", "style", sort_col])

    panels, used_keys = [], set()
    for (g, style), grp in ind.groupby(["group", "style"], sort=False):
        panels.append((style, list(zip(grp["label"], grp["key"])), g or "all"))
        used_keys |= set(grp["key"])

    rest = oth[~oth["key"].isin(used_keys)]
    if len(rest):
        first = rest.sort_values(sort_col if sort_by != "mean" else "label").drop_duplicates("key")
        panels.append(("generalized" if panels else "", list(zip(first["label"], first["key"])), None))
        if panels[:-1]:
            print("NOTE: no human file for AI style(s): " + ", ".join(sorted(first["label"])))

    panels.sort(key=lambda p: PANEL_ORDER.index(p[0]) if p[0] in PANEL_ORDER else len(PANEL_ORDER))   # stable

    grp = pts[pts["kind"] == "group"]                          # group averages (from summarise_group_averages)

    ind_series = list(dict.fromkeys(p[2] for p in panels if p[2]))
    ai_series  = [s for s in sorted(oth["series"].unique()) if s not in ind_series]
    extra_h    = [s for s in dict.fromkeys(grp["series"]) if s in ind_series]   # human series with group averages
    all_series = ind_series + ai_series
    # palette    = sns.color_palette("muted", n_colors=max(len(all_series), 1))
    # colour_of  = {s: palette[i] for i, s in enumerate(all_series)}
    fixed      = {s: c for s, c in {**SERIES_COLORS, **(series_colors or {})}.items() if s in all_series}
    palette    = sns.color_palette("muted", n_colors=len(all_series) + len(fixed))
    free       = [c for c in palette if c not in fixed.values()]      # colours not claimed by a fixed series
    colour_of  = {s: fixed[s] if s in fixed else free.pop(0) for s in all_series}

    max_slots = max(1 + len(ai_series), len(extra_h) + 2 * len(ai_series))
    leg_w     = 3.6
    width     = sum(max(len(p[1]), 1) * (0.42 * max_slots + 0.5) for p in panels) + 0.8 * len(panels) + leg_w
    # panel widths follow cluster counts, except "generalized" gets double its share (others shrink to fit)
    ratios = [float(max(len(p[1]), 1)) for p in panels]
    gi     = [i for i, p in enumerate(panels) if p[0] == "generalized"]
    if gi and len(panels) > 1:
        share    = min(2 * ratios[gi[0]] / sum(ratios), 0.8)
        rest_sum = sum(r for i, r in enumerate(ratios) if i != gi[0])
        ratios   = [share if i == gi[0] else (1 - share) * r / rest_sum for i, r in enumerate(ratios)]
        # shrink the human panels to side_panel_scale of their width; "generalized" keeps its width
        ratios   = [r if i == gi[0] else r * side_panel_scale for i, r in enumerate(ratios)]
        width    = leg_w + (width - leg_w) * sum(ratios) / (share + (1 - share))
    fig, axes = plt.subplots(
        1, len(panels), sharey=True, squeeze=False, figsize=(max(7, width), height),
        gridspec_kw={"width_ratios": ratios},
    )
    axes = axes[0]

    per_file = summarise_by_file(df) if show_files else None
    has_bars = any(p[2] is None for p in panels)
    if has_bars:                                   # shared y range: bars start at the axis floor (or bar_baseline)
        spread = pts[["err", "sd"]].max(axis=1).fillna(0)
        lo, hi = (pts["mean_metric"] - spread).min(), (pts["mean_metric"] + spread).max()
        pad    = 0.06 * (hi - lo if hi > lo else 1.0)
        ylo    = lo - pad if bar_baseline is None else min(lo - pad, bar_baseline)
        bar_bottom = ylo if bar_baseline is None else bar_baseline
        axes[0].set_ylim(ylo, hi + pad)
    kind_order = {"file": 0, "section": 1, "group": 2}
    
    # "generalized" clusters are labelled by the group their averages come from ("non-scifi" -> "fiction")
    group_of = {k.lower(): v for k, v in (GENRE_GROUPS if genre_groups is None else genre_groups).items()}
    for ax, (ptitle, clusters, human_series) in zip(axes, panels):
        slots = ([human_series] if human_series else extra_h) + ai_series
        w = min(0.28, 0.85 / max(len(slots) + (len(ai_series) if human_series is None else 0), 1))
        xtick_pos, xtick_labels = [], []

        for ci, (label, key) in enumerate(clusters):
            cand = pts[(pts["key"] == key) & pts["series"].isin(slots)].copy()
            cand["_o"] = cand["series"].map(slots.index) * 3 + cand["kind"].map(kind_order)
            cand = cand.sort_values("_o")
            for si, (_, r) in enumerate(cand.iterrows()):
                s       = r["series"]
                is_file = r["kind"] == "file"
                is_grp  = r["kind"] == "group"
                x_pos   = ci + (si - (len(cand) - 1) / 2) * w
                if show_files and r["kind"] == "section":
                    sec = f"{r['group']}{SECTION_SEP}{r['style']}" if r["group"] else r["style"]
                    vals = per_file.loc[per_file["section"] == sec, "mean_metric"]
                    jit  = (pd.Series(range(len(vals))) / max(len(vals) - 1, 1) - 0.5) * w * 0.7
                    ax.scatter(x_pos + jit.values, vals, s=26, color=colour_of[s], alpha=0.3, zorder=1)
                if human_series is None:           # bar panel: filled = group average, hollow = style data
                    sd = r["sd"] if pd.notna(r["sd"]) else r["err"]
                    ax.bar(x_pos, r["mean_metric"] - bar_bottom, bottom=bar_bottom, width=w * 0.9,
                           facecolor=colour_of[s] if is_grp else "none", edgecolor=colour_of[s],
                           linewidth=2.2, zorder=2)
                    ax.errorbar(x_pos, r["mean_metric"], yerr=sd, fmt="none", ecolor=colour_of[s],
                                capsize=5, elinewidth=1.8, capthick=1.8, zorder=3)
                    continue
                ax.errorbar(x_pos, r["mean_metric"], yerr=r["err"], fmt="o", color=colour_of[s],
                            markersize=9, capsize=4, linewidth=1.8, markeredgewidth=1.8, zorder=2)

            xtick_pos.append(ci)                 # one label per cluster (author / style), centred under it
            xtick_labels.append(STYLES.get(str(label).strip().lower(), label) if human_series is not None
                                else group_of.get(str(label).strip().lower(), label))
        ax.set_xticks(xtick_pos)
        
        rot = 0 if human_series is None else label_rotation          # "generalized" labels are short: keep them flat
        ax.set_xticklabels(xtick_labels, rotation=rot, ha="center" if rot == 0 else "right")
        ax.set_xlim(-0.5, max(len(clusters), 1) - 0.5)
        ax.xaxis.grid(False)
        for ci in range(len(clusters) - 1):
            ax.axvline(ci + 0.5, color="#CCCCCC", linewidth=1.0, zorder=0)
        ax.set_title(ptitle)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if ax is axes[0] else "")

    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], color=colour_of[s], marker="o", linestyle="none", markersize=12, label=s)
               for s in all_series]
    if has_bars:
        import matplotlib.patches as mpatches
        handles.append(mpatches.Patch(facecolor="#555555", edgecolor="#555555", label="group average (bar)"))
        handles.append(mpatches.Patch(facecolor="none", edgecolor="#555555", linewidth=2, label="style (bar)"))
    fig.legend(handles=handles, loc="center right", frameon=False)

    fig.suptitle(title, y=1.02)
    if xlabel:
        fig.supxlabel(xlabel, fontsize=18, fontweight="bold", color="#000000")
    kind = f"{int(level * 100)}% CI" if interval == "ci" else "SD"
    fig.text(0.99, -0.02, f"human points: own run CI; model points: {kind} across files" + ("; generalized bars: SD" if has_bars else ""), ha="right",
             va="top", fontsize=10, color="#555555")

    fig.tight_layout(rect=(0, 0, 1 - leg_w / max(7, width), 1))
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        duplicate_cleaned(output)
    if show:
        plt.show()

    out = pts.drop(columns=["key"])
    # the returned frame is what gets saved as *_summary.csv: 4 decimal places (plotting used full precision)
    return out.round({c: 4 for c in ("mean_metric", "err", "sd") if c in out.columns})


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(description="Replot a saved results file.")
    ap.add_argument("results_file")
    ap.add_argument("--mode", choices=["section", "file", "sample"], default="section")
    ap.add_argument("--title")
    ap.add_argument("--ylabel", default="Mean metric")
    ap.add_argument("--output", help="image path (default: <results file>_<mode>.png)")
    ap.add_argument("--exclude-prefix")
    ap.add_argument("--interval", choices=["ci", "sd"], default="ci", help="section mode only")
    ap.add_argument("--show-files", action="store_true", help="section mode only")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    out = a.output or f"{os.path.splitext(a.results_file)[0]}_{a.mode}.png"
    extra = {"interval": a.interval, "show_files": a.show_files} if a.mode == "section" else {}
    replot(a.results_file, mode=a.mode, title=a.title, ylabel=a.ylabel, output=out,
           exclude_prefix=a.exclude_prefix, show=not a.no_show, **extra)
    print("saved", out)