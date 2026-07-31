#!/usr/bin/env python3
"""Before editing a file: can this change make results inconsistent?

Answers three questions from facts on disk rather than from judgement:

  1. IS THE CHANGE BEHAVIORAL?  Compares comment- and docstring-stripped fingerprints of the current
     file and a proposed new version. A comment-only edit cannot change a number.
  2. WHO READS THIS FILE RIGHT NOW?  A file in FREEZE_MANIFEST.txt is read by the in-flight batch,
     and a PENDING job reads the code when it STARTS, not when it was submitted. Editing such a file
     splits the batch across two code versions.
  3. WHAT LANDED RESULTS CAME FROM IT?  Directories whose artifacts postdate the file's own mtime
     were produced by the current version; after an edit they are no longer comparable with anything
     produced later.

USAGE
    python3 precheck_code_change.py ags_table5_ablation/run_llm_verifier.py
    python3 precheck_code_change.py <file> <proposed-new-version>     # also classifies the edit

Exit code 1 if the change would split the in-flight batch. CPU only, seconds.
"""
from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "FREEZE_MANIFEST.txt"
LEDGER = ROOT / "ask6_batch_20260730_jobids.txt"
FREEZE_DOC = ROOT / "CODE_FREEZE.md"


def behavioral_fingerprint(path: Path) -> str:
    """Hash of what the file DOES: comments and docstrings removed."""
    text = path.read_text(errors="replace")
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    body = node.body
                    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                            and isinstance(body[0].value.value, str):
                        node.body = body[1:] or [ast.Pass()]
            text = ast.unparse(ast.fix_missing_locations(tree))
        except SyntaxError as exc:
            return f"SYNTAX ERROR: {exc}"
    else:  # shell and everything else: drop whole-line comments and blank lines
        keep = [l for l in text.splitlines()
                if l.strip() and not l.lstrip().startswith("#") or l.lstrip().startswith("#!")]
        text = "\n".join(keep)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def frozen_files() -> dict[str, str]:
    if not MANIFEST.exists():
        return {}
    out = {}
    for line in MANIFEST.read_text().splitlines():
        h, f = line.split(None, 1)
        out[f] = h
    return out


def batch_states() -> list[tuple[str, str, str]]:
    if not LEDGER.exists():
        return []
    rows = []
    for line in LEDGER.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        name, jid = parts
        try:
            r = subprocess.run(["sacct", "-n", "-j", jid, "--format=State%14", "-X"],
                               capture_output=True, text=True, timeout=30)
            state = (r.stdout.strip().splitlines() or ["?"])[0].strip()
        except Exception:
            state = "?"
        rows.append((name, jid, state))
    return rows


def readers_of(rel: str) -> list[str]:
    """Which wrappers/modules mention this file -- a coarse but honest reachability answer."""
    stem = Path(rel).name
    hits = []
    for p in list(ROOT.glob("*.sh")) + list(ROOT.glob("*.py")) + list(ROOT.glob("ags_table5_ablation/*.py")):
        if p.name == stem or p.name == Path(__file__).name:
            continue
        try:
            body = p.read_text(errors="replace")
        except Exception:
            continue
        if stem in body or (p.suffix == ".py" and Path(stem).stem in re.findall(r'(?:import|from)\s+([\w.]+)', body)):
            hits.append(p.name)
    return sorted(hits)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = Path(sys.argv[1])
    if not target.is_absolute():
        target = ROOT / target
    if not target.exists():
        print(f"NEW FILE {target.name}: nothing reads it yet, so it cannot make any existing result "
              f"inconsistent. Safe to add, even during a freeze.")
        return 0
    rel = str(target.relative_to(ROOT))
    proposed = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"=== precheck: {rel} ===\n")

    # ---- 1. behavioral?
    cur = behavioral_fingerprint(target)
    print(f"1. 行为指纹(去注释/docstring 后)")
    print(f"   现在:      {cur}")
    verdict_behavioral = None
    if proposed:
        new = behavioral_fingerprint(proposed)
        print(f"   改动后:    {new}")
        if new.startswith("SYNTAX"):
            print(f"   -> 提案版本无法解析:{new}")
            verdict_behavioral = True
        else:
            verdict_behavioral = new != cur
            print(f"   -> {'行为会变(数字可能变)' if verdict_behavioral else '仅注释/文档改动,数字不会变'}")
    else:
        print("   (没给提案版本,无法判断是否只是注释改动;把新内容写成临时文件再传第二个参数)")

    # ---- 2. who reads it now
    fz = frozen_files()
    in_freeze = rel in fz
    states = batch_states()
    pending = [r for r in states if r[2].upper().startswith(("PENDING", "REQUEUED"))]
    running = [r for r in states if r[2].upper().startswith("RUNNING")]
    print(f"\n2. 谁在读它")
    print(f"   在 FREEZE_MANIFEST 里: {'是' if in_freeze else '否'}"
          + (f"(冻结指纹 {fz[rel]},{'与磁盘一致' if hashlib.sha256(target.read_bytes()).hexdigest()[:16]==fz[rel] else '已与磁盘不一致!'})" if in_freeze else ""))
    print(f"   本批 job: {len(running)} RUNNING, {len(pending)} PENDING, 共 {len(states)}")
    print(f"   引用到该文件的 wrapper/模块: {', '.join(readers_of(rel)) or '(未发现)'}")

    # ---- 3. what landed results came from it
    print(f"\n3. 已经用这一版代码产出的结果")
    mt = target.stat().st_mtime
    produced = []
    for tree in (ROOT / "runs_fintagging_grounding_baseline", ROOT / "runs_ags_verifier_ablation" / "qwen3_32b"):
        if not tree.exists():
            continue
        for d in sorted(tree.iterdir()):
            if not d.is_dir() or d.name.startswith(("_", ".")):
                continue
            newest = max((f.stat().st_mtime for f in d.glob("*") if f.is_file()), default=0)
            if newest > mt:
                produced.append(str(d.relative_to(ROOT)))
    print(f"   文件 mtime 之后有产物写入的目录: {len(produced)}")
    for p in produced[:12]:
        print(f"     {p}")
    if len(produced) > 12:
        print(f"     ... 另外 {len(produced)-12} 个")

    # ---- verdict
    print("\n=== 结论 ===")
    blocking = in_freeze and (pending or running)
    if blocking:
        print(f"   ✗ 会造成不一致。{rel} 属于冻结集,而这批还有 {len(pending)} 个 PENDING、{len(running)} 个 RUNNING。")
        print(f"     PENDING 的 job 是**起跑时**才读代码的,现在改 = 这批劈成两个版本。")
        if verdict_behavioral is False:
            print(f"     即使只改注释:数字不会变,但\"一批一个版本\"这句话就需要解释了 —— {FREEZE_DOC.name} 里禁止。")
        print(f"     做法:记进 TRACK1_ASK6_CHECKLIST.md 的\"下一批\",等 ledger 排空。")
        print(f"     例外门槛:这个改动会让在跑的 job 产出**不可用**的结果(不是\"不够完美\")。")
    elif in_freeze:
        print(f"   ~ 本批已排空,但 {rel} 生成过现有结果。改之前先确认上面第 3 节列的目录里,")
        print(f"     哪些还要被论文引用;改完之后它们与新产物不再可比。")
    else:
        print(f"   ✓ 不在冻结集,本批没有 job 读它。仍然遵守:改完按 verify-by-executing-both-paths 逐分支执行。")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
