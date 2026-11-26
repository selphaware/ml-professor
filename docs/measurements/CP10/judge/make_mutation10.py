"""Make the ninth pass's mutation check from verify/mutation_check.py as it stood: the
forty-four failure modes and sixty-nine small faults, and the newest pass's judge-tools
test left out like the others.

    python -B make_mutation10.py <mutation_check.py as it stood> <out file>
"""
import sys
from pathlib import Path

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
t = SRC.read_bytes().decode("utf-8")


def sub(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:120])
    t = t.replace(old, new)


sub('''The gate's self-test proves the gate against the reference implementation
broken in thirty-seven ways (F1 to F28, and the other forms F6b, F6c, F6d, F7b,
F7c, F7d, F8b, F12b and F14b: D46, D49, D50, D57, D58, D62 to D66, D69 to
D76). This script proves the test suite the same way, against those thirty-seven
and against fifty-one smaller faults no gate check is relied on to catch (M1
to M51; verify/reference/mlprofessor/_variant.py lists them all).''',
    '''The gate's self-test proves the gate against the reference implementation
broken in forty-four ways (F1 to F35, and the other forms F6b, F6c, F6d, F7b,
F7c, F7d, F8b, F12b and F14b: D46, D49, D50, D57, D58, D62 to D66, D69 to
D81). This script proves the test suite the same way, against those forty-four
and against sixty-nine smaller faults no gate check is relied on to catch (M1
to M69; verify/reference/mlprofessor/_variant.py lists them all).''')
sub('''- tests/test_judge_tools.py is not run here (D72). Its tests run the judge's''',
    '''- tests/test_judge_tools.py is not run here (D72), nor is each pass's own file
  of the same kind (tests/test_judge_tools_pass8.py, tests/test_judge_tools_pass9.py:
  D74, D78 point 8). Their tests run the judge's''')
sub('''         + ["F12b", "F14b"] + [f"F{i}" for i in range(15, 29)]
         + [f"M{i}" for i in range(1, 52)])''',
    '''         + ["F12b", "F14b"] + [f"F{i}" for i in range(15, 36)]
         + [f"M{i}" for i in range(1, 70)])''')
sub('''# The strict form of the same, owned by the last task of the eighth pass (D74 point 7).
NOT_RUN_HERE_EITHER = "tests/test_judge_tools_pass8.py"''',
    '''# Each pass's own form of the same (D74 point 7, D78 point 8): the eighth pass's, which
# accepts exactly the failures the ninth pass owns, and the ninth pass's, which is strict.
NOT_RUN_HERE_EITHER = ("tests/test_judge_tools_pass8.py", "tests/test_judge_tools_pass9.py")''')
sub('''           "--ignore", NOT_RUN_HERE, "--ignore", NOT_RUN_HERE_EITHER, "tests"]''',
    '''           "--ignore", NOT_RUN_HERE]
    for other in NOT_RUN_HERE_EITHER:
        cmd += ["--ignore", other]
    cmd += ["tests"]''')
sub('''        print(f"  note: {NOT_RUN_HERE} and {NOT_RUN_HERE_EITHER} are not run here: they run "
              "the judge's other tools on the application")''',
    '''        print(f"  note: {NOT_RUN_HERE} and {', '.join(NOT_RUN_HERE_EITHER)} are not run "
              "here: they run the judge's other tools on the application")''')
DST.write_bytes(t.encode("utf-8"))
print("wrote", DST)
