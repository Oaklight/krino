# Sanitized historical harness. The identifier lists are examples:
# replace them with your operator-local names before running.
import json
import re
import subprocess

PATTERN = re.compile(
    r"/home/|/homes/|H100|H200|A100"
)
NUMBERS = [
    268, 264, 216, 217, 145, 49,
    208, 219, 229, 230, 234, 225, 210, 209, 220, 205, 211, 218, 195, 223,
    174, 186, 176, 185, 184, 226,
]


def api(path):
    result = subprocess.run(
        ["gh", "api", path], capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def scan(number):
    out = []
    pr = api(f"repos/Oaklight/krino/pulls/{number}")
    issue = api(f"repos/Oaklight/krino/issues/{number}")
    body = (issue or {}).get("body") or ""
    if PATTERN.search(body):
        out.append({"where": "body", "author": issue["user"]["login"], "id": None})
    for comment in api(f"repos/Oaklight/krino/issues/{number}/comments") or []:
        if PATTERN.search(comment.get("body") or ""):
            out.append(
                {
                    "where": "issue_comment",
                    "author": comment["user"]["login"],
                    "id": comment["id"],
                }
            )
    if pr:
        for review in api(f"repos/Oaklight/krino/pulls/{number}/reviews") or []:
            if PATTERN.search(review.get("body") or ""):
                out.append(
                    {
                        "where": "review",
                        "author": review["user"]["login"],
                        "id": review["id"],
                        "state": review["state"],
                    }
                )
        for comment in api(f"repos/Oaklight/krino/pulls/{number}/comments") or []:
            if PATTERN.search(comment.get("body") or ""):
                out.append(
                    {
                        "where": "review_comment",
                        "author": comment["user"]["login"],
                        "id": comment["id"],
                    }
                )
    return out


report = {}
for number in NUMBERS:
    locations = scan(number)
    if locations:
        report[number] = locations
print(json.dumps(report, indent=1))
