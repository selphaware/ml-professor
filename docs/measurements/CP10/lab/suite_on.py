"""CP10: the existing suite against a package (the prototype, or the reference as it
stands), in a throwaway project outside the repo, as verify/mutation_check.py builds one.

    python -B suite_on.py <package folder> <project folder> [--parts 6] [--tests <tests folder>]

Writes <project>/part<k>.junit.xml and prints the failing tests. The two judge-tool
files are left out: they run the judge's tools on the application.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(r"E:\CODE\ml-professor")
LEFT_OUT = ("test_judge_tools.py", "test_judge_tools_pass8.py")


def ignore(folder, names):
    return [n for n in names if n in ("__pycache__", ".pytest_cache")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package")
    parser.add_argument("project")
    parser.add_argument("--parts", type=int, default=6)
    parser.add_argument("--tests", default=str(REPO / "tests"))
    parser.add_argument("--only", default=None, help="comma-separated test files")
    args = parser.parse_args()
    project = Path(args.project)
    if project.exists():
        shutil.rmtree(project)
    shutil.copytree(args.tests, project / "tests", ignore=ignore)
    shutil.copytree(args.package, project / "mlprofessor", ignore=ignore)
    for name in ("README.md", "requirements.txt"):
        shutil.copyfile(REPO / name, project / name)
    shutil.copytree(REPO / "examples", project / "examples")
    files = sorted(p.name for p in (project / "tests").glob("test_*.py") if p.name not in LEFT_OUT)
    if args.only:
        files = [f for f in files if f in args.only.split(",")]
    # the slow files first, one to a part, then the rest dealt round
    sizes = {f: (project / "tests" / f).stat().st_size for f in files}
    heavy = [f for f in files if f.startswith("test_type_matrix")]
    rest = sorted((f for f in files if f not in heavy), key=lambda f: -sizes[f])
    parts = [[] for _ in range(args.parts)]
    for i, f in enumerate(heavy + rest):
        parts[i % args.parts].append(f)
    env = dict(os.environ)
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONSTARTUP", "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")

    def run(k):
        report = project / f"part{k}.junit.xml"
        cmd = [sys.executable, "-B", "-m", "pytest", "-c", "tests/pytest.ini", "--rootdir", ".",
               "--confcutdir", "tests", "-p", "no:cacheprovider", "-q", "--junitxml", str(report)] \
            + [f"tests/{f}" for f in parts[k]]
        proc = subprocess.run(cmd, cwd=str(project), env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        (project / f"part{k}.out.txt").write_text(proc.stdout[-200000:] + proc.stderr[-20000:],
                                                  encoding="utf-8")
        return proc.returncode

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.parts) as pool:
        codes = list(pool.map(run, range(args.parts)))
    failed, total = [], 0
    for k in range(args.parts):
        report = project / f"part{k}.junit.xml"
        if not report.is_file():
            print(f"part {k}: no report (exit {codes[k]})")
            continue
        for item in ET.parse(report).iter("testcase"):
            total += 1
            kinds = {child.tag for child in item}
            if kinds & {"failure", "error", "skipped"}:
                module = item.get("classname", "").split(".")[-1]
                text = ""
                for child in item:
                    if child.tag in ("failure", "error"):
                        text = (child.get("message") or "")[:300].replace("\n", " ")
                failed.append((f"{module}.py::{item.get('name')}", text))
    print(f"{total} cases, {len(failed)} failing, exit codes {codes}, {time.time() - started:.0f} s")
    for name, text in sorted(failed):
        print(f"FAILED {name}\n       {text}")


if __name__ == "__main__":
    main()
