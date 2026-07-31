#!/usr/bin/env python3
"""Number of table rows in a .tex still carrying a -- placeholder cell. Used by the watcher
to decide when it is done; prints a single integer so the shell can compare it."""
import re
import sys

lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
spans, stack = {}, []
for i, line in enumerate(lines):
    s = line.strip()
    if s.startswith(r"\begin{table"):
        stack.append(i)
    m = re.search(r"\\label\{(tab:[^}]+)\}", s)
    if m and stack:
        spans[m.group(1)] = (stack[-1], i)
    if s.startswith(r"\end{table") and stack:
        stack.pop()

def is_placeholder(cell: str) -> bool:
    """A cell holding nothing but a -- placeholder, with or without the \\ph colour wrapper.

    Placeholder cells are written \\ph{--} so they show red in the draft; matching only the
    bare -- would report 0 for a table that is entirely unfilled.
    """
    c = cell.strip()
    c = re.sub(r"^\\(?:ph|upd)\{(.*)\}$", r"\1", c).strip()
    return c == "--"


n = 0
for a, b in spans.values():
    for j in range(a, b):
        row = lines[j].rstrip()
        if row.endswith(r"\\") and any(is_placeholder(c) for c in row.removesuffix(r"\\").split("&")):
            n += 1
print(n)
