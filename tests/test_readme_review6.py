"""T59: the README on the seventh pass (spec 1; D69 to D72).

It judges `README.md`, which the builder owns; it neither edits it nor adds files. The
README is read as `test_readme_examples` reads it.
"""
import test_readme_examples as rx


def test_the_readme_names_what_the_seventh_pass_added():
    text = rx._readme()
    for needed in ("fast_profile_group_too_large", "99991231", "YYYYMMDDHH"):
        assert needed in text, f"the README does not hold {needed!r}"
    found = rx._UNDER_JUDGE.findall(text)
    assert not found, f"the README names a path under the judge's folders: {found}"
