"""Bring the words on coded copies up to the reference's own 30-seed numbers (CP10,
outr2/b_bands.json): under the random scheme 16 bands and up are excluded on every table;
beside entities on most; under time order on the walks no count of eight or fewer is
warned about or excluded, more bands only sometimes; on a time-ordered target that does
not wander (`flat_tail`) as under the random scheme."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def edit(name, pairs):
    p = HERE / name
    t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert t.count(old) == 1, (name, t.count(old), old[:90])
        t = t.replace(old, new)
    p.write_text(t, encoding="utf-8", newline="\n")


edit("edit_spec10a.py", [
    ('''headline at 0.95 to 1.0 (the seventh review, finding 1), and a numeric code
of the target likewise, in both tasks. Read by its levels it is excluded from
16 bands up, under the random and the grouped scheme; coarser bands are
weaker columns, and section 11 says what becomes of them (D78 point 4, D80
point 3).''',
     '''headline at 0.95 to 1.0 (the seventh review, finding 1), and a numeric code of
the target likewise, in both tasks. Read by its levels it is excluded on
every table from 16 bands up under the random scheme, and on most beside
entities (10.11); coarser bands are weaker columns, and under time order the
latest rows' levels are not the earlier rows': section 11 says what becomes
of them (D78 point 4, D80 point 3).'''),
    ('''| the target cut into 16 or more bands, as text with labels in any order or as a numeric code, beside the target (random or grouped scheme) | excluded as a suspected leak: read by its levels (step 8; D77 point 3); coarser bands as section 11 says (D80 point 3) |''',
     '''| the target cut into 16 or more bands, as text with labels in any order or as a numeric code (random scheme) | excluded as a suspected leak: read by its levels (step 8; D77 point 3); beside entities, under time order and with coarser bands, as section 11 says (D80 point 3) |'''),
])
edit("edit_spec10b.py", [
    ('''  `headline.rests_on` when they lift the headline above the floor (on 134 to
  180 of 180 tables under the random scheme). **Under time order no band
  count is warned about or excluded:** the levels of the latest rows are not
  those of the earlier ones. Wherever a banded copy was neither excluded,
  warned about nor named, it lifted the headline by less than 0.05: at most
  +0.037 over about 1,500 tables (10.11). From 16 bands up such a copy is
  excluded under the random and the grouped scheme.''',
     '''  `headline.rests_on` when they lift the headline above the floor (on 134 to
  180 of 180 tables under the random scheme). **Under time order, on a target
  that wanders, no count of eight bands or fewer is warned about or
  excluded,** since the levels of the latest rows are not those of the
  earlier ones: on the gate's `random_walk`, `int_stamps` and `panel`, none
  of 2 to 30 bands was excluded and at most 3 of 90 warned about; at 40 and
  60 bands some are warned about (up to 51 of 90) and a few excluded (3 of
  90). On a time-ordered target that does not wander (`flat_tail`) the copy
  is read as under the random scheme: 8 and 16 bands warned about, 24 and up
  excluded. Wherever a banded copy was neither excluded, warned about nor
  named, it lifted the headline by less than 0.05: at most +0.037 over about
  1,500 tables (10.11). From 16 bands up such a copy is excluded on every
  table under the random scheme, and beside entities on most: 87 to 90 of 90
  beside a quarter entity, 33 to 90 of 90 beside a dominant one.'''),
])
edit("edit_spec10c.py", [
    ('''  87 and 3, 60 bands 90 excluded. Under time order (`random_walk`,
  `int_stamps`, `panel`) no band count was excluded or warned about, and
  wherever a banded copy was named by nothing it lifted the headline by at
  most +0.019.''',
     '''  87 and 3, 60 bands 90 excluded. Under time order on the walks
  (`random_walk`, `int_stamps`, `panel`, 90 tables each count): 8 to 30
  bands excluded on none and warned about on 0 to 3, named in the list on 98
  to 150 of 270; 40 bands warned about on 24; 60 bands warned about on 87 and
  excluded on 3. Wherever a banded copy on a walk was named by nothing it
  lifted the headline by at most +0.019. On `flat_tail`, a time-ordered
  target that does not wander, 8 and 16 bands were warned about on 90 of 90
  and 24 and up excluded on 90 of 90.'''),
])
edit("edit_claude10.py", [
    ('''classification, 10 of each class or all of one. So the target cut into 16 or
more bands, with labels in any order, is excluded; six to eight bands are
warned about, and four or fewer read like an honest column and are only
named (D80 point 3).''',
     '''classification, 10 of each class or all of one. So the target cut into 16 or
more bands, with labels in any order, is excluded under the random scheme;
six to eight bands are warned about, and four or fewer read like an honest
column and are only named (D80 point 3). Under time order on a wandering
target the latest rows' levels are new, and a coded copy is mostly neither
excluded nor warned about (spec 11).'''),
])
print("edited")
