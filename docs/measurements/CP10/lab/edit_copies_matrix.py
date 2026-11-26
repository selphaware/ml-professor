r"""edit_make_proto5.py: R2 as the owner worded it, copies "as the models receive them".

Two features are copies when the parts of the final fit's matrix they give are the same
in every training row and every holdout row: for numeric features, their one column each,
agreeing to within 1e-9 (the columns are imputed and scaled, so a positive rescaling of a
column is a copy of it); for categorical features, their blocks of level columns, holding
the same columns in any order (so a relabelling of the levels is a copy). Features of
different kinds are never copies. `MLP_A_COPIES=values` keeps the earlier rule: the same
value in every such row, as a step of the seam is handed them.
"""
from pathlib import Path

p = Path(__file__).resolve().parent / "edit_make_proto5.py"
t = p.read_text(encoding="utf-8")
old = t[t.index("        # D78 (3): two features are copies"):t.index("        copy_sets = list(copy_sets.values())\n")
                                                              + len("        copy_sets = list(copy_sets.values())\n")]
new = r'''        # D78 (3): two features are copies when the parts of the final fit's matrix they
        # give are the same in every training row and every holdout row: a numeric
        # feature's column agreeing to within COPY_TOLERANCE, a categorical feature's
        # block of level columns holding the same columns in any order
        if P.A_COPIES == "values":
            reach = rows[np.concatenate([train, holdout])]
            copy_sets = {}
            for name, values in prepared.items():
                held = values[reach]
                if kinds[name] == "numeric":
                    held = np.where(np.isnan(held) | (np.abs(held) > TOO_LARGE), np.nan,
                                    held) + 0.0
                    key = ("numeric", held.tobytes())
                else:
                    key = ("categorical", "\\x00".join(held.astype(str)))
                copy_sets.setdefault(key, []).append(name)
            copy_sets = list(copy_sets.values())
        else:
            whole = np.vstack([x_fit, x_holdout])
            head = min(16, len(whole))

            def block(name):
                part = whole[:, blocks[name]]
                if kinds[name] == "numeric":
                    return part
                return part[:, np.lexsort(part[::-1])] if part.shape[1] else part

            seen_blocks = {name: block(name) for name in prepared}

            def alike(a, b):
                x, z = seen_blocks[a], seen_blocks[b]
                if kinds[a] != kinds[b] or x.shape != z.shape:
                    return False
                if kinds[a] == "numeric":
                    return bool(np.all(np.abs(x[:head] - z[:head]) <= P.COPY_TOLERANCE)
                                and np.all(np.abs(x - z) <= P.COPY_TOLERANCE))
                return bool(np.array_equal(x[:head], z[:head]) and np.array_equal(x, z))

            copy_sets, placed = [], set()
            for name in prepared:
                if name in placed:
                    continue
                members = [name] + [other for other in prepared
                                    if other != name and other not in placed
                                    and P.A_COPIES != "none" and alike(name, other)]
                placed.update(members)
                copy_sets.append(members)
'''
t = t.replace(old, new)
p.write_text(t, encoding="utf-8", newline="\n")
q = Path(__file__).resolve().parent / "_proto.py"
s = q.read_text(encoding="utf-8")
s = s.replace('A_COPIES = _get("MLP_A_COPIES", "1") == "1"',
              'A_COPIES = _get("MLP_A_COPIES", "matrix")    # matrix | values | none\n'
              '# D78 (3): two numeric columns of the matrix agree when within this of each other\n'
              'COPY_TOLERANCE = float(_get("MLP_COPY_TOLERANCE", "1e-9"))')
assert "COPY_TOLERANCE" in s
q.write_text(s, encoding="utf-8", newline="\n")
print("edited")
