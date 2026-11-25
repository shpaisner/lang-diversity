import argparse, copy, json, os, pathlib, re, shutil, time
from dataclasses import dataclass, field
import requests

API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE = "https://openrouter.ai/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

AIGEN_ROOT = pathlib.Path("../story/aigen")      # every model gets its own folder under here
TERMINAL_BAD = ("failed", "expired", "cancelled")

N_PER_STYLE = 10
MIN_WORDS = 1000
MAX_ROUNDS = 3            # retry rounds after the first submission


# ---------- models: add an entry here to query another model ----------

@dataclass(frozen=True)
class ModelConfig:
    slug: str                                   # OpenRouter batch model slug
    dir: str                                    # folder name under AIGEN_ROOT (what the plot scripts see as the model)
    body: dict = field(default_factory=dict)    # extra parameters merged into every request body

    def matches(self, api_model):
        """Does the `model` a batch reports (e.g. 'google/gemini-4-hydrogen-20261112') belong to this config?"""
        base = re.escape(self.slug.split(":")[0])
        return bool(re.fullmatch(base + r"(?:-\d{8})?(?::.*)?", api_model or ""))


MODELS = {
    "sonnet": ModelConfig(
        slug="anthropic/claude-sonnet-5.5:batch",
        dir="sonnet5-5",
        body={"max_tokens": 8000},                           # sonnet does not expose a thinking level
    ),
    "gemini": ModelConfig(
        slug="google/gemini-4-hydrogen:batch",
        dir="gemini-4-hydrogen",
        body={"max_tokens": 24000, "reasoning": {"effort": "medium"}},   # gemini uses more thinking tokens
    ),
}


# ---------- prompts ----------

STYLES = {
    # science fiction authors
    "verne": "Jules Verne",
    "kipling" : "Rudyard Kipling",
    "wells" : "H. G. Wells",
    "lovecraft": "H.P. Lovecraft",
    "bradbury": "Ray Bradbury",
    # non-science fiction writers
    "james": "Henry James (1843-1916)",
    "doyle": "Arthur Conan Doyle",
    "woolf": "Virginia Woolf",
    "joyce": "James Joyce",
    "hemingway": "Ernest Hemingway",
    # genre specific prompts
    "scifi": "a science fiction author",
    "non-scifi": "a non-science fiction author"
}

BASE_PROMPT = """Continue the following story based on H. G. Wells "In the Days of the Comet": 
William Leadford is a poor man from Staffordshire in love with Nettie, who runs off with Verrall. 
Meanwhile, a comet is approaching Earth, and Europe is drifting toward war."""

LENGTH_RULES = """Length requirements:
- Write between 1,800 and 2,000 words.
- Structure the story in 6 to 7 scenes of roughly 250-300 words each.
- Do not summarize or rush the ending; give the final scene its full length.
- Output only the story text, with no title, preamble, or word count."""


# ---------- helpers that don't depend on the model ----------

def get_batch(batch_id):
    """One status check. Returns the batch dict, or None if it isn't visible yet (404)."""
    r = requests.get(f"{BASE}/batches/{batch_id}", headers=HEADERS)
    if r.status_code == 404:
        return None
    if not r.ok:
        print(r.status_code, r.text)
        r.raise_for_status()
    return r.json()


def clean_story(text):
    # Remove scene-break lines: 3+ of * - or _ (optionally spaced), alone on a line
    text = re.sub(r"(?m)^[ \t]*(?:[*_-][ \t]*){3,}$", "", text)
    text = text.replace("*", "")                  # strip italics/bold markers
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse the gaps the breaks leave
    return text.strip()


def base_id(custom_id):
    return re.sub(r"__r\d+$", "", custom_id)     # kipling__01__r2 -> kipling__01


def process_results(batch):
    """Returns (good, bad, raw_ids). raw_ids: base id -> custom_id as submitted."""
    good, bad, raw_ids = {}, {}, {}
    for res in batch.get("results", []):
        cid = base_id(res["custom_id"])
        raw_ids[cid] = res["custom_id"]
        resp = res.get("response") or {}
        if res.get("error") or resp.get("status_code") != 200:
            bad[cid] = ("error", str(res.get("error") or resp.get("status_code")), "")
            continue
        choice = resp["body"]["choices"][0]
        text = clean_story(choice["message"].get("content") or "")
        words = len(text.split())
        if choice.get("finish_reason") != "stop":
            bad[cid] = ("truncated", f"finish_reason={choice.get('finish_reason')}", text)
        elif words < MIN_WORDS:
            bad[cid] = ("short", f"{words} words", text)
        else:
            good[cid] = text
    return good, bad, raw_ids


def make_retry(orig_req, kind, round_no):
    r = copy.deepcopy(orig_req)
    r["custom_id"] = f"{orig_req['custom_id']}__r{round_no}"
    body = r["body"]
    if kind == "truncated":
        body["max_tokens"] += 4000
    elif kind == "short":
        body["messages"][0]["content"] += (
            "\n\nIMPORTANT: a previous attempt came out too short. "
            "Write at least 1,500 words."
        )
    return r


# ---------- everything that touches a model's folder lives on Target ----------

class Target:
    """
    One model and its folder. Requests, state, raw results, manifest and story files all
    live under out_dir, and every path below is derived from it, so nothing for one model
    can end up in another model's folder.
    """
    def __init__(self, key):
        self.key = key
        self.cfg = MODELS[key]
        self.out_dir = AIGEN_ROOT / self.cfg.dir
        self.batch_dir = self.out_dir / "_batches"
        self.state_file = self.out_dir / "state.json"
        self._next_index = {}                    # (folder, style) -> next file number

    def say(self, msg):
        print(f"[{self.key}] {msg}")

    # ----- requests -----
    def build_requests(self, styles=None, per_style=None):
        reqs = []
        for key in (styles or STYLES):
            prompt = (
                f"{BASE_PROMPT}\n\n"
                f"Write in the prose style of {STYLES[key]}.\n"
                f"{LENGTH_RULES}\n"
            )
            for i in range(per_style or N_PER_STYLE):
                body = {"messages": [{"role": "user", "content": prompt}]}
                body.update(copy.deepcopy(self.cfg.body))
                reqs.append({"custom_id": f"{key}__{i:02d}", "body": body})
        return reqs

    def submit_batch(self, reqs):
        # endpoint and model must come before requests in the JSON body
        payload = {"endpoint": "/v1/chat/completions", "model": self.cfg.slug, "requests": reqs}
        r = requests.post(f"{BASE}/batches", headers=HEADERS, data=json.dumps(payload))
        if not r.ok:
            self.say(f"{r.status_code} {r.text}")
            r.raise_for_status()
        return r.json()

    # ----- state -----
    def load_state(self):
        if self.state_file.exists():
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        return {"batches": {}, "jobs": {}}

    def save_state(self, state):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.state_file)                      # atomic swap

    def log_batch(self, batch, round_no, n_requests):
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "batches.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "batch_id": batch["id"], "round": round_no, "model": self.cfg.slug,
                "n_requests": n_requests, "submitted_at": int(time.time()),
                "results_file": str(self.batch_dir / f"{batch['id']}.json"),
            }) + "\n")

    def register_batch(self, state, batch, job_id, round_no, reqs):
        """Save the exact requests sent (needed to build retries later) and track the batch."""
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        req_file = self.batch_dir / f"{batch['id']}.requests.json"
        req_file.write_text(json.dumps(reqs), encoding="utf-8")
        state["batches"][batch["id"]] = {
            "job": job_id, "round": round_no, "n_requests": len(reqs),
            "status": "pending", "requests_file": str(req_file), "model": self.cfg.slug,
        }
        self.log_batch(batch, round_no, len(reqs))

    # ----- writing story files -----
    def _scan_next_index(self, folder, style):
        pattern = re.compile(rf"^{re.escape(style)}__(\d+)\.txt$")
        highest = -1
        if folder.exists():
            for p in folder.iterdir():
                m = pattern.match(p.name)
                if m:
                    highest = max(highest, int(m.group(1)))
        return highest + 1

    def write_story(self, cid, text, folder=None, source=None):
        style = cid.split("__")[0]
        d = self.out_dir / (folder or style)
        d.mkdir(parents=True, exist_ok=True)

        key = (str(d), style)
        if key not in self._next_index:
            self._next_index[key] = self._scan_next_index(d, style)

        while True:
            n = self._next_index[key]
            self._next_index[key] += 1
            path = d / f"{style}__{n:02d}.txt"
            try:
                with open(path, "x", encoding="utf-8") as f:
                    f.write(text)
                break
            except FileExistsError:
                continue

        record = {"file": str(path), "base_custom_id": cid, "words": len(text.split()),
                  "model": self.cfg.slug}
        record.update(source or {})
        with open(self.out_dir / "manifest.jsonl", "a", encoding="utf-8") as m:
            m.write(json.dumps(record) + "\n")
        return path

    # ----- submit -----
    def submit(self, styles=None, per_style=None):
        state = self.load_state()
        reqs = self.build_requests(styles, per_style)
        assert reqs and len(reqs) <= 5000, "Need 1-5000 requests per batch"
        batch = self.submit_batch(reqs)
        state["jobs"][batch["id"]] = {
            "rounds": 0,
            "stories": {r["custom_id"]: {"state": "pending", "batch_id": batch["id"]} for r in reqs},
        }
        self.register_batch(state, batch, batch["id"], 0, reqs)
        self.save_state(state)
        self.say(f"submitted {batch['id']} ({len(reqs)} requests, model {self.cfg.slug})")
        self.say(f"results will be written to {self.out_dir}")

    # ----- status -----
    def status(self):
        state = self.load_state()
        if not state["batches"]:
            self.say("no batches recorded")
        for bid, b in state["batches"].items():
            if b["status"] != "pending":
                self.say(f"{bid}  round {b['round']}  {b['status']}")
                continue
            info = get_batch(bid)
            if info is None:
                self.say(f"{bid}  round {b['round']}  not visible yet")
            else:
                self.say(f"{bid}  round {b['round']}  {info['status']}  {info.get('request_counts')}")

    # ----- processing finished batches -----
    def process_batch(self, state, bid, b, final):
        """Write stories from a finished batch and record each story's outcome."""
        job = state["jobs"][b["job"]]
        results_file = self.batch_dir / f"{bid}.json"
        results_file.write_text(json.dumps(final, indent=2), encoding="utf-8")
        sent = json.loads(pathlib.Path(b["requests_file"]).read_text(encoding="utf-8"))
        requested = {base_id(r["custom_id"]) for r in sent}

        if final["status"] != "completed":                      # failed / expired / cancelled
            for cid in requested:
                job["stories"][cid] = {"state": "flagged", "kind": f"batch_{final['status']}",
                                       "detail": str(final.get("error")), "batch_id": bid, "raw_id": None}
            b["status"] = final["status"]
            return

        good, bad, raw_ids = process_results(final)
        for cid in sorted(requested):
            if cid in good:
                self.write_story(cid, good[cid], source={
                    "batch_id": bid, "round": b["round"],
                    "custom_id": raw_ids.get(cid), "results_file": str(results_file)})
                job["stories"][cid] = {"state": "good", "batch_id": bid}
            else:
                kind, detail, _ = bad.get(cid, ("missing", "no result returned", ""))
                job["stories"][cid] = {"state": "flagged", "kind": kind, "detail": detail,
                                       "batch_id": bid, "raw_id": raw_ids.get(cid)}
        b["status"] = "processed"

    def abandon(self, state, job, cid, s):
        """Give up on a flagged story: keep its best attempt in _below_minimum."""
        text = ""
        rf = self.batch_dir / f"{s['batch_id']}.json"
        if rf.exists():
            _, bad, _ = process_results(json.loads(rf.read_text(encoding="utf-8")))
            text = bad.get(cid, ("", "", ""))[2]
        if text:
            self.write_story(cid, text, folder="_below_minimum", source={
                "batch_id": s["batch_id"], "round": state["batches"][s["batch_id"]]["round"],
                "custom_id": s.get("raw_id"), "results_file": str(rf)})
        job["stories"][cid]["state"] = "abandoned"
        self.say(f"STILL BAD: {cid} ({s['kind']}: {s['detail']})")

    def retry_or_finalize(self, state, job_id, retry=True):
        job = state["jobs"][job_id]
        if any(b["job"] == job_id and b["status"] == "pending" for b in state["batches"].values()):
            return                                              # still waiting on a batch
        flagged = {cid: s for cid, s in job["stories"].items() if s["state"] == "flagged"}
        if not flagged or not retry:
            if flagged:
                self.say(f"job {job_id[-6:]}: {len(flagged)} flagged (retry disabled)")
            return

        sent = json.loads(pathlib.Path(state["batches"][job_id]["requests_file"]).read_text(encoding="utf-8"))
        originals = {r["custom_id"]: r for r in sent}
        can_retry = {cid: s for cid, s in flagged.items() if (originals.get(cid) or {}).get("body")}
        for cid, s in flagged.items():                          # no request body to resend
            if cid not in can_retry:
                self.say(f"{cid}: original request unknown, cannot retry")
                self.abandon(state, job, cid, s)

        if not can_retry:
            return
        if job["rounds"] < MAX_ROUNDS:
            round_no = job["rounds"] + 1
            retries = [make_retry(originals[cid], s["kind"], round_no) for cid, s in can_retry.items()]
            batch = self.submit_batch(retries)                  # same model as the original job
            job["rounds"] = round_no
            for cid in can_retry:
                job["stories"][cid] = {"state": "pending", "batch_id": batch["id"]}
            self.register_batch(state, batch, job_id, round_no, retries)
            self.say(f"job {job_id[-6:]}: submitted retry round {round_no} "
                     f"({len(retries)} stories) as {batch['id']}")
        else:                                                   # out of retries
            for cid, s in can_retry.items():
                self.abandon(state, job, cid, s)

    def fetch(self, retry=True, check_model=True):
        state = self.load_state()
        if not any(b["status"] == "pending" for b in state["batches"].values()):
            self.say("no pending batches")

        for bid, b in list(state["batches"].items()):
            if b["status"] != "pending":
                continue
            final = get_batch(bid)
            if final is None:
                self.say(f"{bid}: not visible yet, try again shortly")
            elif final["status"] == "completed" or final["status"] in TERMINAL_BAD:
                api_model = final.get("model")
                if check_model and api_model and not self.cfg.matches(api_model):
                    self.say(f"{bid}: REFUSING to write results. The batch ran {api_model!r} but this "
                             f"folder is for {self.cfg.slug!r}. Check how it was recorded "
                             f"(or rerun with --ignore-model-check).")
                    continue
                self.process_batch(state, bid, b, final)
                self.save_state(state)
                self.say(f"{bid}: {final['status']}, cost ${(final.get('usage') or {}).get('cost') or 0:.4f}")
            else:
                self.say(f"{bid}: still {final['status']} {final.get('request_counts')}")

        for job_id in list(state["jobs"]):
            self.retry_or_finalize(state, job_id, retry)
            self.save_state(state)

        for job_id, job in state["jobs"].items():
            counts = {}
            for s in job["stories"].values():
                counts[s["state"]] = counts.get(s["state"], 0) + 1
            self.say(f"job {job_id[-6:]}: {counts}")

    # ----- adopting batches that predate the state file -----
    def adopt(self, items, retry=True, force=False, check_model=True):
        """items: [(batch_id, info)]. Start tracking them under this model's folder, then fetch.
        Retries are rebuilt from the CURRENT build_requests(), so they only match the original
        request if STYLES, the prompt and this model's body haven't changed."""
        state = self.load_state()
        mf = self.out_dir / "manifest.jsonl"
        in_manifest = set()
        if mf.exists():
            in_manifest = {json.loads(l).get("batch_id")
                           for l in mf.read_text(encoding="utf-8").splitlines() if l.strip()}
        current = {r["custom_id"]: r for r in self.build_requests()}

        for bid, info in items:
            if bid in state["batches"]:
                self.say(f"{bid}: already tracked, skipping")
                continue
            if bid in in_manifest and not force:
                self.say(f"{bid}: stories from this batch are already in the manifest "
                         f"(use --force to write them again)")
                continue

            if info["status"] == "completed":
                ids = sorted({base_id(r["custom_id"]) for r in info.get("results", [])})
            else:                                               # still running: assume the current set
                ids = list(current)
                total = (info.get("request_counts") or {}).get("total")
                if total is not None and total != len(ids):
                    self.say(f"{bid}: WARNING batch has {total} requests but build_requests() makes {len(ids)}")
            reqs = [current.get(cid) or {"custom_id": cid, "body": None} for cid in ids]
            unknown = [cid for cid in ids if cid not in current]
            if unknown:
                self.say(f"{bid}: {len(unknown)} custom_id(s) not in the current build_requests() "
                         f"(e.g. {unknown[:3]}); they can be processed but not retried")

            state["jobs"][bid] = {
                "rounds": 0,
                "stories": {r["custom_id"]: {"state": "pending", "batch_id": bid} for r in reqs},
            }
            self.register_batch(state, info, bid, 0, reqs)
            self.say(f"{bid}: adopted ({len(reqs)} requests, status {info['status']}) -> {self.out_dir}")
        self.save_state(state)
        self.fetch(retry=retry, check_model=check_model)

    # ----- loose batch result files (no state needed) -----
    def done_batch_ids(self):
        mf = self.out_dir / "manifest.jsonl"
        if not mf.exists():
            return set()
        return {json.loads(l).get("batch_id") for l in mf.read_text().splitlines() if l.strip()}

    def reprocess(self, paths, force=False):
        """Rebuild story files from saved batch result JSONs. No API calls.
        Each batch file is independent output; only retry results (custom_id ending
        in __rN) replace an earlier flagged result for the same story.
        Batches already in the manifest are skipped unless force=True."""
        already = set() if force else self.done_batch_ids()
        batches = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                b = json.load(f)
            if b.get("status") != "completed":
                self.say(f"skipping {p}: status={b.get('status')}")
            elif b["id"] in already:
                self.say(f"skipping {p}: {b['id']} already in manifest (use --force to redo)")
            elif b.get("model") and not self.cfg.matches(b["model"]):
                self.say(f"skipping {p}: it ran {b['model']!r}, not {self.cfg.slug!r}")
            else:
                batches.append((p, b))
        batches.sort(key=lambda pb: pb[1].get("created_at", 0))   # oldest first

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        stories, leftovers, provenance = {}, {}, {}      # all keyed by (batch_id, custom_id)

        for round_no, (path, b) in enumerate(batches):
            results_file = self.batch_dir / f"{b['id']}.json"
            if pathlib.Path(path).resolve() != results_file.resolve():
                shutil.copyfile(path, results_file)

            good, bad, raw_ids = process_results(b)
            for cid in set(good) | set(bad):
                key = (b["id"], cid)
                if raw_ids.get(cid, cid) != cid:             # a retry: replaces its flagged original
                    for old in [k for k in leftovers if k[1] == cid]:
                        del leftovers[old]
                provenance[key] = {
                    "batch_id": b["id"], "round": round_no,
                    "custom_id": raw_ids.get(cid), "results_file": str(results_file),
                }
                if cid in good:
                    stories[key] = good[cid]
                else:
                    leftovers[key] = bad[cid]
            self.say(f"{path}: {len(good)} good, {len(bad)} flagged")

        for key in sorted(stories):
            self.write_story(key[1], stories[key], source=provenance[key])
        for key in sorted(leftovers):
            kind, detail, text = leftovers[key]
            if text:
                self.write_story(key[1], text, folder="_below_minimum", source=provenance[key])
            self.say(f"STILL BAD: {key[1]} ({kind}: {detail}) in {key[0]}")
        self.say(f"{len(stories)} stories written from {len(batches)} batch file(s)")


# ---------- commands that span models ----------

def pick_targets(keys):
    """The models a command applies to: those named, or all of them."""
    return [Target(k) for k in (keys or MODELS)]


def cmd_models():
    for key, cfg in MODELS.items():
        t = Target(key)
        pending = sum(1 for b in t.load_state()["batches"].values() if b["status"] == "pending")
        print(f"{key:8} {cfg.slug:38} -> {t.out_dir}   ({pending} pending)")


def cmd_adopt(batch_ids, model_key=None, retry=True, force=False, check_model=True):
    """Work out which model each batch belongs to (from what the API says it ran), then adopt it there."""
    by_model = {}
    for bid in batch_ids:
        info = get_batch(bid)
        if info is None:
            print(f"{bid}: not found (wrong ID, different API key, or older than 30 days?)")
            continue
        detected = [k for k, c in MODELS.items() if c.matches(info.get("model"))]
        if model_key:
            if model_key not in detected and not force:
                print(f"{bid}: REFUSING. It ran {info.get('model')!r}, which doesn't match "
                      f"--model {model_key} ({MODELS[model_key].slug!r}). Use --force to override.")
                continue
            key = model_key
        elif len(detected) == 1:
            key = detected[0]
        else:
            print(f"{bid}: can't tell which model this is (API says {info.get('model')!r}). Pass --model.")
            continue
        by_model.setdefault(key, []).append((bid, info))
    for key, items in by_model.items():
        Target(key).adopt(items, retry=retry, force=force, check_model=check_model)


def main():
    keys = sorted(MODELS)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="submit one batch per chosen model, then exit")
    s.add_argument("--model", nargs="+", required=True, choices=keys)
    s.add_argument("--styles", nargs="+", choices=sorted(STYLES), help="only these styles (default all)")
    s.add_argument("--per-style", type=int, help=f"stories per style (default {N_PER_STYLE})")

    st = sub.add_parser("status", help="check pending batches without processing")
    st.add_argument("--model", nargs="+", choices=keys, help="default: all models")

    f = sub.add_parser("fetch", help="process finished batches; submit retries")
    f.add_argument("--model", nargs="+", choices=keys, help="default: all models")
    f.add_argument("--no-retry", action="store_true", help="don't submit retry batches")
    f.add_argument("--ignore-model-check", action="store_true")

    a = sub.add_parser("adopt", help="track batches submitted before state.json existed, then fetch")
    a.add_argument("batch_ids", nargs="+")
    a.add_argument("--model", choices=keys, help="default: detected from the batch")
    a.add_argument("--no-retry", action="store_true")
    a.add_argument("--force", action="store_true", help="adopt even if the manifest already has it / the model differs")
    a.add_argument("--ignore-model-check", action="store_true")

    r = sub.add_parser("reprocess", help="rebuild story files from saved batch result JSON files")
    r.add_argument("--model", required=True, choices=keys)
    r.add_argument("files", nargs="+")
    r.add_argument("--force", action="store_true")

    sub.add_parser("models", help="list configured models and where their results go")
    args = ap.parse_args()

    if args.cmd == "submit":
        for t in pick_targets(args.model):
            t.submit(args.styles, args.per_style)
        print("Safe to close this terminal. Later run:  python openrouter.py fetch")
    elif args.cmd == "status":
        for t in pick_targets(args.model):
            t.status()
    elif args.cmd == "fetch":
        for t in pick_targets(args.model):
            t.fetch(retry=not args.no_retry, check_model=not args.ignore_model_check)
    elif args.cmd == "adopt":
        cmd_adopt(args.batch_ids, args.model, retry=not args.no_retry, force=args.force,
                  check_model=not args.ignore_model_check)
    elif args.cmd == "reprocess":
        Target(args.model).reprocess(args.files, force=args.force)
    else:
        cmd_models()


if __name__ == "__main__":
    main()