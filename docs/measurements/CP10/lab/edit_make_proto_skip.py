"""One edit of make_proto.py: a third way with a level not seen in a fold's training
rows (`MLP_B_UNSEEN=skip`): its rows are not scored for the level reader."""
from pathlib import Path

p = Path(__file__).resolve().parent / "make_proto.py"
t = p.read_text(encoding="utf-8")


def sub(old, new):
    global t
    assert t.count(old) == 1, (t.count(old), old[:90])
    t = t.replace(old, new)


sub('''    readers = [("tree", lambda base: guess)]
    if other:
        line = np.concatenate(other)''',
    '''    readers = [("tree", lambda base: guess)]
    only = {}
    if other:
        line = np.concatenate(other)''')
sub('''            readers.append(("levels", from_levels))
    eligible = None''',
    '''            readers.append(("levels", from_levels))
            if P.B_UNSEEN == "skip":
                only[from_levels] = seen
    eligible = None''')
sub('''                own_errors = (truth - reader(base_pred)) ** 2
                keep = _scored_rows(own_errors, base_error, eligible) if unscored \\\\
                    else np.ones(len(truth), dtype=bool)''',
    '''                own_errors = (truth - reader(base_pred)) ** 2
                may = eligible
                if reader in only:
                    if only[reader].sum() < 20:
                        return -1.0     # too few rows of a level seen: no reading by levels
                    own_errors, base_error = own_errors[only[reader]], base_error[only[reader]]
                    may = None if eligible is None else eligible[only[reader]]
                keep = _scored_rows(own_errors, base_error, may) if unscored \\\\
                    else np.ones(len(own_errors), dtype=bool)''')
p.write_text(t, encoding="utf-8", newline="\n")
print("edited", p)
