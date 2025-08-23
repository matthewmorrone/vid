import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from index import parse_time_spec, human_size


def test_parse_time_spec():
    assert parse_time_spec("middle", 100.0) == 50.0
    assert parse_time_spec("25%", 200.0) == 50.0
    assert parse_time_spec("25%", None) == 1.0
    assert parse_time_spec("10", None) == 10.0
    assert parse_time_spec("bad", 0) == 1.0


def test_human_size():
    assert human_size(0) == "0.0B"
    assert human_size(1024) == "1.0KB"
    assert human_size(1024 * 1024) == "1.0MB"
    assert human_size(-1) == "?"
