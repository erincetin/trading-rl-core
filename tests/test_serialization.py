from trading_rl.experiment.serialization import make_json_safe


def _fn():
    return 1


def test_make_json_safe_converts_types_and_callables():
    payload = {
        "t": int,
        "f": _fn,
        "lst": [str, _fn],
        "tpl": (float, _fn),
    }
    out = make_json_safe(payload)
    assert out["t"] == "int"
    assert out["f"] == "_fn"
    assert out["lst"] == ["str", "_fn"]
    assert out["tpl"] == ["float", "_fn"]
