# Sanitized historical harness. The identifier lists are examples:
# replace them with your operator-local names before running.
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DRY = "--apply" not in sys.argv
BACKUP = Path("/tmp/text_backup")
BACKUP.mkdir(exist_ok=True)

NUMBERS = sorted(set([
    6, 15, 19, 23, 25, 26, 27, 28, 29, 30, 31, 34, 35, 37, 38, 39, 40, 41,
    42, 43, 47, 49, 50, 52, 56, 61, 62, 69, 73, 85, 91, 94, 95, 97, 100,
    104, 108, 111, 115, 116, 124, 127, 139, 144, 145, 159, 160, 161, 167,
    169, 174, 176, 181, 184, 185, 186, 192, 194, 195, 205, 208, 209, 210,
    211, 217, 218, 219, 220, 223, 224, 225, 226, 229, 230, 234, 245, 248,
    251, 252, 258, 261, 262, 264, 266, 268,
]))

EDITABLE = {"example-org", "example-bot[bot]"}

REDACTIONS = [
    (re.compile(r"/mnt/example-storage/[^ ]*/checkout"), "<shared-storage>/checkout"),
    (re.compile(r"/mnt/example-storage/[^ ]*"), "<shared-storage>"),
    (re.compile(r"/home/[^/]+/projects/example-repo"), "<checkout>"),
    (re.compile(r"/home/[^/]+"), "~"),
    (re.compile(r"/homes/[^/]+"), "~"),
    (re.compile(r"/homes/"), "/users/"),
]



def api(path):
    result = subprocess.run(["gh", "api", path], capture_output=True, text=True)
    return json.loads(result.stdout) if result.returncode == 0 else None


def transform(text):
    original = text
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    text = re.sub(
        r"(^|(?<=[.!?])[ \t]+)host ([ABCD])\b",
        lambda m: f"{m.group(1)}Host {m.group(2)}",
        text,
        flags=re.M,
    )
    return text, text != original


def collect(number):
    items = []
    issue = api(f"repos/Oaklight/krino/issues/{number}")
    if not issue:
        return items
    items.append(("body", number, None, issue["user"]["login"], issue.get("body") or ""))
    for comment in api(f"repos/Oaklight/krino/issues/{number}/comments") or []:
        items.append(("issue_comment", number, comment["id"], comment["user"]["login"], comment.get("body") or ""))
    pr = api(f"repos/Oaklight/krino/pulls/{number}")
    if pr:
        for review in api(f"repos/Oaklight/krino/pulls/{number}/reviews") or []:
            items.append(("review", number, review["id"], review["user"]["login"], review.get("body") or ""))
        for comment in api(f"repos/Oaklight/krino/pulls/{number}/comments") or []:
            items.append(("review_comment", number, comment["id"], comment["user"]["login"], comment.get("body") or ""))
    return items


report = []
for number in NUMBERS:
    for kind, n, item_id, author, body in collect(number):
        if not body:
            continue
        new, changed = transform(body)
        if not changed:
            continue
        report.append(
            {
                "number": n,
                "kind": kind,
                "id": item_id,
                "author": author,
                "old": body,
                "new": new,
            }
        )

print(json.dumps([{k: v for k, v in r.items() if k not in ("old", "new")} for r in report], indent=1))
print("changed", len(report))

if DRY:
    PREVIEW = Path("/tmp/redact_preview")
    PREVIEW.mkdir(exist_ok=True)
    for r in report:
        stem = f"{r['kind']}_{r['number']}_{r['id']}"
        (PREVIEW / f"{stem}.old.md").write_text(r["old"])
        (PREVIEW / f"{stem}.new.md").write_text(r["new"])

if not DRY:
    wrote = 0
    for entry in report:
        kind, n, item_id = entry["kind"], entry["number"], entry["id"]
        if entry["author"] not in EDITABLE:
            print("SKIP", kind, n, item_id, entry["author"])
            continue
        (BACKUP / f"{kind}_{n}_{item_id}.txt").write_text(entry["old"])
        path = BACKUP / f"{kind}_{n}_{item_id}.new.txt"
        path.write_text(entry["new"])
        if kind == "body":
            cmd = ["gh", "api", "-X", "PATCH", f"repos/Oaklight/krino/issues/{n}", "-F", f"body=@{path}"]
        elif kind == "issue_comment":
            cmd = ["gh", "api", "-X", "PATCH", f"repos/Oaklight/krino/issues/comments/{item_id}", "-F", f"body=@{path}"]
        elif kind == "review_comment":
            cmd = ["gh", "api", "-X", "PATCH", f"repos/Oaklight/krino/pulls/comments/{item_id}", "-F", f"body=@{path}"]
        else:
            node = subprocess.run(
                ["gh", "api", f"repos/Oaklight/krino/pulls/{n}/reviews/{item_id}", "--jq", ".node_id"],
                capture_output=True, text=True,
            ).stdout.strip()
            if entry["author"] != "example-bot[bot]" or not node:
                print("SKIP review", n, item_id, entry["author"])
                continue
            body_json = json.dumps(entry["new"])
            query = (
                'mutation { updatePullRequestReview(input: {pullRequestReviewId: "%s", body: %s})'
                " { pullRequestReview { id } } }" % (node, body_json)
            )
            cmd = ["gh", "api", "graphql", "-f", f"query={query}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        status = "ok" if result.returncode == 0 else "FAIL"
        print(status, kind, n, item_id)
        wrote += 1
    print("wrote", wrote)
