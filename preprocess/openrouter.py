import os, re, time, json, copy, pathlib, requests, shutil
import argparse   # add to your imports if it isn't there already

API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE = "https://openrouter.ai/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
MODEL = "anthropic/claude-sonnet-5.5:batch"
# MODEL = "google/gemini-4-hydrogen:batch"

N_PER_STYLE = 10
MIN_WORDS = 1000
MAX_ROUNDS = 3            # retry rounds after the first submission
# OUT_DIR = pathlib.Path("../story/aigen/gemini-4-hydrogen")
OUT_DIR = pathlib.Path("../story/aigen/sonnet5-5")

STYLES = {
    # science fiction authors
    # "verne": "Jules Verne",
    # "kipling" : "Rudyard Kipling",
    # "wells" : "H. G. Wells",
    # "lovecraft": "H.P. Lovecraft",
    # "bradbury": "Ray Bradbury",
    # non-science fiction writers
    "james": "Henry James (1843-1916)",
    # "doyle": "Arthur Conan Doyle",
    # "woolf": "Virginia Woolf",
    # "joyce": "James Joyce",
    # "hemingway": "Ernest Hemingway",
    # # genre specific prompts
    # "scifi": "a science fiction author",
    # "non-scifi": "a non-science fiction author"
}

BASE_PROMPT = """Continue the following story based on H. G. Wells "In the Days of the Comet": 
William Leadford is a poor man from Staffordshire in love with Nettie, who runs off with Verrall. 
Meanwhile, a comet is approaching Earth, and Europe is drifting toward war."""

LENGTH_RULES = """Length requirements:
- Write between 1,800 and 2,000 words.
- Structure the story in 6 to 7 scenes of roughly 250-300 words each.
- Do not summarize or rush the ending; give the final scene its full length.
- Output only the story text, with no title, preamble, or word count."""

def build_requests():
    reqs = []
    for key, author in STYLES.items():
            prompt = (
                f"{BASE_PROMPT}\n\n"
                f"Write in the prose style of {author}.\n"
                f"{LENGTH_RULES}\n"
            )
            for i in range(0,N_PER_STYLE):
                reqs.append({
                    "custom_id": f"{key}__{i:02d}",
                    "body": {
                        "messages": [{"role": "user", "content": prompt}],
                        # "max_tokens": 24000,
                        # "reasoning": {"effort": "medium"},
                    },
                })
    return reqs


def submit_batch(reqs):
    # endpoint and model must come before requests in the JSON body
    payload = {
        "endpoint": "/v1/chat/completions",
        "model": MODEL,
        "requests": reqs,
    }
    r = requests.post(f"{BASE}/batches", headers=HEADERS, data=json.dumps(payload))
    if not r.ok:
        print(r.status_code, r.text)  # shows which field was rejected
        r.raise_for_status()
    return r.json()

BATCH_DIR = OUT_DIR / "_batches"

def log_batch(batch, round_no, n_requests):
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "batches.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "batch_id": batch["id"],
            "round": round_no,
            "model": MODEL,
            "n_requests": n_requests,
            "submitted_at": int(time.time()),
            "results_file": str(BATCH_DIR / f"{batch['id']}.json"),
        }) + "\n")

def wait_for_batch(batch_id, poll_seconds=30, not_found_grace=120):
    start = time.time()
    while True:
        r = requests.get(f"{BASE}/batches/{batch_id}", headers=HEADERS)

        if r.status_code == 404 and time.time() - start < not_found_grace:
            print("404 (may not be visible yet):", r.text)
            time.sleep(10)
            continue
        if not r.ok:
            print(r.status_code, r.text)  # the body should say what's wrong
            r.raise_for_status()

        batch = r.json()
        status = batch["status"]
        print("status:", status, batch.get("request_counts"))
        if status in ("completed", "failed", "expired", "cancelled"):
            return batch
        time.sleep(poll_seconds)

# ---------- processing results ----------

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

# ---------- writing files ----------

_next_index = {}   # (folder, style) -> next number, cached after first scan


def _scan_next_index(folder, style):
    pattern = re.compile(rf"^{re.escape(style)}__(\d+)\.txt$")
    highest = -1
    if folder.exists():
        for p in folder.iterdir():
            m = pattern.match(p.name)
            if m:
                highest = max(highest, int(m.group(1)))
    return highest + 1


def write_story(cid, text, folder=None, source=None):
    style = cid.split("__")[0]
    d = OUT_DIR / (folder or style)
    d.mkdir(parents=True, exist_ok=True)

    key = (str(d), style)
    if key not in _next_index:
        _next_index[key] = _scan_next_index(d, style)

    while True:
        n = _next_index[key]
        _next_index[key] += 1
        path = d / f"{style}__{n:02d}.txt"
        try:
            with open(path, "x", encoding="utf-8") as f:
                f.write(text)
            break
        except FileExistsError:
            continue

    record = {"file": str(path), "base_custom_id": cid, "words": len(text.split())}
    record.update(source or {})
    with open(OUT_DIR / "manifest.jsonl", "a", encoding="utf-8") as m:
        m.write(json.dumps(record) + "\n")
    return path


# ---------- main ----------


def done_batch_ids():
    """Batch IDs already recorded in the manifest."""
    mf = OUT_DIR / "manifest.jsonl"
    if not mf.exists():
        return set()
    return {json.loads(l).get("batch_id") for l in mf.read_text().splitlines() if l.strip()}


def reprocess(paths, force=False):
    """Rebuild story files from saved batch result JSONs. No API calls.
    Each batch file is independent output; only retry results (custom_id ending
    in __rN) replace an earlier flagged result for the same story.
    Batches already in the manifest are skipped unless force=True."""
    already = set() if force else done_batch_ids()
    batches = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            b = json.load(f)
        if b.get("status") != "completed":
            print(f"skipping {p}: status={b.get('status')}")
        elif b["id"] in already:
            print(f"skipping {p}: {b['id']} already in manifest (use force=True to redo)")
        else:
            batches.append((p, b))
    batches.sort(key=lambda pb: pb[1].get("created_at", 0))   # oldest first

    OUT_DIR.mkdir(exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    stories, leftovers, provenance = {}, {}, {}      # all keyed by (batch_id, custom_id)

    for round_no, (path, b) in enumerate(batches):
        results_file = BATCH_DIR / f"{b['id']}.json"
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
        print(f"{path}: {len(good)} good, {len(bad)} flagged")

    for key in sorted(stories):
        write_story(key[1], stories[key], source=provenance[key])
    for key in sorted(leftovers):
        kind, detail, text = leftovers[key]
        if text:
            write_story(key[1], text, folder="_below_minimum", source=provenance[key])
        print(f"STILL BAD: {key[1]} ({kind}: {detail}) in {key[0]}")

    print(f"{len(stories)} stories written from {len(batches)} batch file(s)")





# ---------- resumable workflow: submit / status / fetch ----------

STATE_FILE = OUT_DIR / "state.json"
TERMINAL_BAD = ("failed", "expired", "cancelled")


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"batches": {}, "jobs": {}}


def save_state(state):
    OUT_DIR.mkdir(exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)                      # atomic swap


def get_batch(batch_id):
    """One status check. Returns the batch dict, or None if it isn't visible yet (404)."""
    r = requests.get(f"{BASE}/batches/{batch_id}", headers=HEADERS)
    if r.status_code == 404:
        return None
    if not r.ok:
        print(r.status_code, r.text)
        r.raise_for_status()
    return r.json()


def register_batch(state, batch, job_id, round_no, reqs):
    """Save the exact requests sent (needed to build retries later) and track the batch."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    req_file = BATCH_DIR / f"{batch['id']}.requests.json"
    req_file.write_text(json.dumps(reqs), encoding="utf-8")
    state["batches"][batch["id"]] = {
        "job": job_id, "round": round_no, "n_requests": len(reqs),
        "status": "pending", "requests_file": str(req_file),
    }
    log_batch(batch, round_no, len(reqs))


def cmd_submit():
    state = load_state()
    reqs = build_requests()
    assert reqs and len(reqs) <= 5000, "Need 1-5000 requests per batch"
    batch = submit_batch(reqs)
    state["jobs"][batch["id"]] = {
        "rounds": 0,
        "stories": {r["custom_id"]: {"state": "pending", "batch_id": batch["id"]} for r in reqs},
    }
    register_batch(state, batch, batch["id"], 0, reqs)
    save_state(state)
    print(f"Submitted {batch['id']} ({len(reqs)} requests).")
    print("Safe to close this terminal. Later run:  python <script>.py fetch")


def cmd_status():
    state = load_state()
    if not state["batches"]:
        print("No batches recorded.")
    for bid, b in state["batches"].items():
        if b["status"] != "pending":
            print(f"{bid}  round {b['round']}  {b['status']}")
            continue
        info = get_batch(bid)
        if info is None:
            print(f"{bid}  round {b['round']}  not visible yet")
        else:
            print(f"{bid}  round {b['round']}  {info['status']}  {info.get('request_counts')}")


def process_batch(state, bid, b, final):
    """Write stories from a finished batch and record each story's outcome."""
    job = state["jobs"][b["job"]]
    results_file = BATCH_DIR / f"{bid}.json"
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
            write_story(cid, good[cid], source={
                "batch_id": bid, "round": b["round"],
                "custom_id": raw_ids.get(cid), "results_file": str(results_file)})
            job["stories"][cid] = {"state": "good", "batch_id": bid}
        else:
            kind, detail, _ = bad.get(cid, ("missing", "no result returned", ""))
            job["stories"][cid] = {"state": "flagged", "kind": kind, "detail": detail,
                                   "batch_id": bid, "raw_id": raw_ids.get(cid)}
    b["status"] = "processed"


def abandon(state, job, cid, s):
    """Give up on a flagged story: keep its best attempt in _below_minimum."""
    text = ""
    rf = BATCH_DIR / f"{s['batch_id']}.json"
    if rf.exists():
        _, bad, _ = process_results(json.loads(rf.read_text(encoding="utf-8")))
        text = bad.get(cid, ("", "", ""))[2]
    if text:
        write_story(cid, text, folder="_below_minimum", source={
            "batch_id": s["batch_id"], "round": state["batches"][s["batch_id"]]["round"],
            "custom_id": s.get("raw_id"), "results_file": str(rf)})
    job["stories"][cid]["state"] = "abandoned"
    print(f"STILL BAD: {cid} ({s['kind']}: {s['detail']})")


def retry_or_finalize(state, job_id, retry=True):
    job = state["jobs"][job_id]
    if any(b["job"] == job_id and b["status"] == "pending" for b in state["batches"].values()):
        return                                              # still waiting on a batch
    flagged = {cid: s for cid, s in job["stories"].items() if s["state"] == "flagged"}
    if not flagged or not retry:
        if flagged:
            print(f"job {job_id[-6:]}: {len(flagged)} flagged (retry disabled)")
        return

    sent = json.loads(pathlib.Path(state["batches"][job_id]["requests_file"]).read_text(encoding="utf-8"))
    originals = {r["custom_id"]: r for r in sent}
    can_retry = {cid: s for cid, s in flagged.items() if (originals.get(cid) or {}).get("body")}
    for cid, s in flagged.items():                          # no request body to resend
        if cid not in can_retry:
            print(f"{cid}: original request unknown, cannot retry")
            abandon(state, job, cid, s)

    if not can_retry:
        return
    if job["rounds"] < MAX_ROUNDS:
        round_no = job["rounds"] + 1
        retries = [make_retry(originals[cid], s["kind"], round_no) for cid, s in can_retry.items()]
        batch = submit_batch(retries)
        job["rounds"] = round_no
        for cid in can_retry:
            job["stories"][cid] = {"state": "pending", "batch_id": batch["id"]}
        register_batch(state, batch, job_id, round_no, retries)
        print(f"job {job_id[-6:]}: submitted retry round {round_no} "
              f"({len(retries)} stories) as {batch['id']}")
    else:                                                   # out of retries
        for cid, s in can_retry.items():
            abandon(state, job, cid, s)


def cmd_adopt(batch_ids, retry=True, force=False):
    """Start tracking batches submitted before state.json existed, then run a fetch.
    Retries are rebuilt from the CURRENT build_requests() (STYLES, prompt, N_PER_STYLE),
    so they only match the original request if those haven't changed."""
    state = load_state()
    in_manifest = set()
    mf = OUT_DIR / "manifest.jsonl"
    if mf.exists():
        in_manifest = {json.loads(l).get("batch_id")
                       for l in mf.read_text(encoding="utf-8").splitlines() if l.strip()}
    current = {r["custom_id"]: r for r in build_requests()}

    for bid in batch_ids:
        if bid in state["batches"]:
            print(f"{bid}: already tracked, skipping")
            continue
        if bid in in_manifest and not force:
            print(f"{bid}: stories from this batch are already in the manifest "
                  f"(use --force to write them again)")
            continue
        info = get_batch(bid)
        if info is None:
            print(f"{bid}: not found (wrong ID, different API key, or older than 30 days?)")
            continue

        if info["status"] == "completed":
            ids = sorted({base_id(r["custom_id"]) for r in info.get("results", [])})
        else:                                               # still running: assume the current set
            ids = list(current)
            total = (info.get("request_counts") or {}).get("total")
            if total is not None and total != len(ids):
                print(f"{bid}: WARNING batch has {total} requests but build_requests() makes {len(ids)}")
        reqs = [current.get(cid) or {"custom_id": cid, "body": None} for cid in ids]
        unknown = [cid for cid in ids if cid not in current]
        if unknown:
            print(f"{bid}: {len(unknown)} custom_id(s) not in the current build_requests() "
                  f"(e.g. {unknown[:3]}); they can be processed but not retried")

        state["jobs"][bid] = {
            "rounds": 0,
            "stories": {r["custom_id"]: {"state": "pending", "batch_id": bid} for r in reqs},
        }
        register_batch(state, info, bid, 0, reqs)
        print(f"{bid}: adopted ({len(reqs)} requests, status {info['status']})")
    save_state(state)
    cmd_fetch(retry=retry)

def cmd_fetch(retry=True):
    state = load_state()
    if not any(b["status"] == "pending" for b in state["batches"].values()):
        print("No pending batches.")

    for bid, b in list(state["batches"].items()):
        if b["status"] != "pending":
            continue
        final = get_batch(bid)
        if final is None:
            print(f"{bid}: not visible yet, try again shortly")
        elif final["status"] == "completed" or final["status"] in TERMINAL_BAD:
            process_batch(state, bid, b, final)
            save_state(state)
            print(f"{bid}: {final['status']}, cost ${(final.get('usage') or {}).get('cost') or 0:.4f}")
        else:
            print(f"{bid}: still {final['status']} {final.get('request_counts')}")

    for job_id in list(state["jobs"]):
        retry_or_finalize(state, job_id, retry)
        save_state(state)

    for job_id, job in state["jobs"].items():
        counts = {}
        for s in job["stories"].values():
            counts[s["state"]] = counts.get(s["state"], 0) + 1
        print(f"job {job_id[-6:]}: {counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("submit", help="submit a new batch and exit")
    sub.add_parser("status", help="check pending batches without processing")
    f = sub.add_parser("fetch", help="process finished batches; submit retries")
    f.add_argument("--no-retry", action="store_true", help="don't submit retry batches")
    a = sub.add_parser("adopt", help="track batches submitted before state.json existed, then fetch")
    a.add_argument("batch_ids", nargs="+")
    a.add_argument("--no-retry", action="store_true")
    a.add_argument("--force", action="store_true", help="adopt even if the manifest already has this batch")
    args = ap.parse_args()

    if args.cmd == "submit":
        cmd_submit()
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "adopt":
        cmd_adopt(args.batch_ids, retry=not args.no_retry, force=args.force)
    else:
        cmd_fetch(retry=not args.no_retry)
