import pytest

from alhambra_mixes.printing import _format_title


def test_title():
    assert _format_title("title name µ", 1, "orgtbl") == "* title name µ"
