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

        df = load_file_data("results/lex_20261120.txt", exclude_prefix="break_")
        plot_by_file(df, title="Lexical diversity per file", output="results/lex.png")
"""

import re
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
            "axes.edgecolor":     "#CCCCCC",
            "axes.linewidth":     0.8,
            "axes.spines.top":    False,
            "axes.spines.bottom": False,
            "axes.spines.left":   False,
            "axes.spines.right":  False,
            "grid.color":         "#EEEEEE",
            "grid.linewidth":     0.8,
            "axes.titlesize":     13,
            "axes.titleweight":   "bold",
            "axes.labelsize":     11,
            "axes.labelcolor":    "#333333",
            "xtick.labelsize":    9,
            "ytick.labelsize":    9,
            "xtick.color":        "#555555",
            "ytick.color":        "#555555",
            "xtick.major.size":   0,
            "ytick.major.size":   3,
            "text.color":         "#333333",
            "figure.titlesize":   14,
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


def summarise_points(df: pd.DataFrame, individual_groups=("human",),
                     interval: str = "ci", level: float = 0.95) -> pd.DataFrame:
    """
    One row per plotted point.
      - sections whose group is in individual_groups -> one point PER FILE
        (err = the file's own run-based CI, as in the old per-file plot)
      - all other sections -> one point per section, aggregated across files
        (err = CI or SD across files, see summarise_by_section)

    Returns columns: group, style, label, mean_metric, err, n_files, kind ("file"/"section")
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
        pf["kind"]    = "file"
        parts.append(pf[["group", "style", "label", "mean_metric", "err", "n_files", "kind"]])

    if agg:
        ag = summarise_by_section(df[df["section"].isin(agg)], interval=interval, level=level).copy()
        ag["group"] = ag["section"].map(lambda s: _split_section(s)[0])
        ag["style"] = ag["section"].map(lambda s: _split_section(s)[1])
        ag["label"] = ag["style"]
        ag["kind"]  = "section"
        parts.append(ag[["group", "style", "label", "mean_metric", "err", "n_files", "kind"]])

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
    figsize:  tuple = (10, 5),
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
                    color=colour, linewidth=1.2, zorder=1)
            ax.scatter(data["sample_size"], data["mean_metric"],
                       s=30, color=colour, marker=marker, zorder=2, label=fname)
            if "wse" in data.columns:
                ax.errorbar(
                    data["sample_size"], data["mean_metric"],
                    yerr=data["wse"], fmt="none", color=colour, capsize=3, linewidth=1
                )

        # legend: file colours + section shape patches
        handles, labels = ax.get_legend_handles_labels()
        if has_sections:
            import matplotlib.lines as mlines
            for section, marker in section_marker.items():
                handles.append(mlines.Line2D([], [], color="grey", marker=marker,
                                             linestyle="none", label=f"[{section}]"))
        ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    else:
        ax.plot(summary["sample_size"], summary["mean_metric"],
                color=palette[0], linewidth=1.2, zorder=1)
        ax.scatter(summary["sample_size"], summary["mean_metric"],
                   s=30, color=palette[0], zorder=2)
        if "wse" in summary.columns:
            ax.errorbar(
                summary["sample_size"], summary["mean_metric"],
                yerr=summary["wse"], fmt="none", color=palette[0], capsize=3, linewidth=1
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

    if show:
        plt.show()


def plot_by_file(
    df:       pd.DataFrame,
    title:    str = "Mean metric per file",
    xlabel:   str = None,
    ylabel:   str = "Mean metric",
    output:   str = None,
    height:   float = 5,
    aspect:   float = 0.8,
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
            sns.scatterplot(data=data, x="filename", y="mean_metric", s=30, ax=ax)
            if "wse" in data.columns:
                ax.errorbar(
                    x_pos, data["mean_metric"],
                    yerr=data["wse"], fmt="none", color="steelblue", capsize=3, linewidth=1
                )
            ax.set_title(section)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel if ax == axes[0] else "")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(data["filename"], rotation=45, ha="right")

        fig.suptitle(title, y=1.02)
        if xlabel:
            fig.supxlabel(xlabel, fontsize=11, color="#333333")
        plt.tight_layout()
        # fig.subplots_adjust(wspace=0.05)
        if output:
            fig.savefig(output, dpi=150, bbox_inches="tight")
        
        if show:
            plt.show()

    else:
        fig, ax = plt.subplots(figsize=(max(6, len(summary) * 0.6), height))
        x_pos = range(len(summary))
        sns.scatterplot(data=summary, x="filename", y="mean_metric", s=30, ax=ax)
        if "wse" in summary.columns:
            ax.errorbar(
                x_pos, summary["mean_metric"],
                yerr=summary["wse"], fmt="none", color="steelblue", capsize=3, linewidth=1
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
    height:            float = 5,
    show:              bool  = True,
) -> pd.DataFrame:
    """
    Clusters by style name, one panel per human style directory.

      * One PANEL per human directory ("scifi", "fiction").
      * Inside a panel, one CLUSTER per human file.
      * Each cluster holds one point per SERIES: the human file itself ("human")
        plus every model's point for the style directory with the same name
        (matched case-insensitively). Series are colour-coded, and every point gets
        its own angled x-axis label (with file count included for AI series).
      * AI styles that match no human file are collected in an extra "no human match" panel.
    """
    set_theme()
    pts = summarise_points(df, individual_groups, interval, level)
    pts["group"]  = pts["group"].fillna("")
    pts["key"]    = pts["label"].str.strip().str.lower()
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
        panels.append(("no human match" if panels else "", list(zip(first["label"], first["key"])), None))
        if panels[:-1]:
            print("NOTE: no human file for AI style(s): " + ", ".join(sorted(first["label"])))

    ind_series = list(dict.fromkeys(p[2] for p in panels if p[2]))
    ai_series  = [s for s in sorted(oth["series"].unique()) if s not in ind_series]
    all_series = ind_series + ai_series
    palette    = sns.color_palette("muted", n_colors=max(len(all_series), 1))
    colour_of  = {s: palette[i] for i, s in enumerate(all_series)}

    max_slots = 1 + len(ai_series)
    leg_w     = 1.7
    width     = sum(max(len(p[1]), 1) * (0.38 * max_slots + 0.5) for p in panels) + 0.8 * len(panels) + leg_w
    fig, axes = plt.subplots(
        1, len(panels), sharey=True, squeeze=False, figsize=(max(7, width), height),
        gridspec_kw={"width_ratios": [max(len(p[1]), 1) for p in panels]},
    )
    axes = axes[0]

    per_file = summarise_by_file(df) if show_files else None
    for ax, (ptitle, clusters, human_series) in zip(axes, panels):
        slots = ([human_series] if human_series else []) + ai_series
        w = min(0.28, 0.85 / max(len(slots), 1))
        xtick_pos, xtick_labels = [], []

        for ci, (label, key) in enumerate(clusters):
            present = [s for s in slots if ((pts["series"] == s) & (pts["key"] == key)).any()]
            for si, s in enumerate(present):
                off  = (si - (len(present) - 1) / 2) * w
                rows = pts[(pts["series"] == s) & (pts["key"] == key)]
                for _, r in rows.iterrows():
                    is_file = r["kind"] == "file"
                    x_pos   = ci + off
                    if show_files and not is_file:
                        sec = f"{r['group']}{SECTION_SEP}{r['style']}" if r["group"] else r["style"]
                        vals = per_file.loc[per_file["section"] == sec, "mean_metric"]
                        jit  = (pd.Series(range(len(vals))) / max(len(vals) - 1, 1) - 0.5) * w * 0.7
                        ax.scatter(x_pos + jit.values, vals, s=12, color=colour_of[s], alpha=0.3, zorder=1)
                    ax.errorbar(x_pos, r["mean_metric"], yerr=r["err"], fmt="o", color=colour_of[s],
                                markersize=5 if is_file else 6, capsize=3, linewidth=1.2, zorder=2)

                    pt_label = r["label"] if is_file else f"{r["label"]} (n={int(r['n_files'])})"
                    xtick_pos.append(x_pos)
                    xtick_labels.append(pt_label)

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, rotation=label_rotation,
                           ha="center" if label_rotation == 0 else "right")
        ax.set_xlim(-0.5, max(len(clusters), 1) - 0.5)
        ax.xaxis.grid(False)
        for ci in range(len(clusters) - 1):
            ax.axvline(ci + 0.5, color="#E3E3E3", linewidth=0.8, zorder=0)
        ax.set_title(ptitle)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel if ax is axes[0] else "")

    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], color=colour_of[s], marker="o", linestyle="none", label=s)
               for s in all_series]
    fig.legend(handles=handles, loc="center right", frameon=False)

    fig.suptitle(title, y=1.02)
    if xlabel:
        fig.supxlabel(xlabel, fontsize=11, color="#333333")
    kind = f"{int(level * 100)}% CI" if interval == "ci" else "SD"
    fig.text(0.99, -0.02, f"human points: own run CI; model points: {kind} across files", ha="right",
             va="top", fontsize=7, color="#888888")

    fig.tight_layout(rect=(0, 0, 1 - leg_w / max(7, width), 1))
    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return pts.drop(columns=["key"])


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