"""Every front door returns the same result: Python API, run folder, tool registry, command line, MCP server."""

import contextlib
import io
import json

import pytest

from horizoneval import evaluate, tools
from horizoneval.cli import main
from horizoneval.frame import read
from horizoneval.result import RESULT_VERSION, dumps, to_result, validate

from test_frame import PERIOD, nixtla_frames


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("frames")
    fc, train = nixtla_frames()
    fc.to_csv(d / "cv.csv", index=False)
    train.to_parquet(d / "train.parquet", index=False)
    return d


def _args(d, **extra):
    return {"forecasts": str(d / "cv.csv"), "train": str(d / "train.parquet"), "period": PERIOD,
            "reference": "Naive", **extra}


def _cli(argv) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(argv)
    return buf.getvalue().strip()


def test_every_path_returns_the_same_document(files, tmp_path):
    python = dumps(to_result(evaluate(read(files / "cv.csv"), read(files / "train.parquet"), period=PERIOD,
                                      reference="Naive")))
    registry = tools.call_json("score", _args(files, out=str(tmp_path / "run")))
    folder = (tmp_path / "run" / "result.json").read_text().strip()
    cli = _cli(["score", str(files / "cv.csv"), "--train", str(files / "train.parquet"), "--period", str(PERIOD),
                "--reference", "Naive", "--out", str(tmp_path / "cli"), "--json"])
    assert python == registry == folder == cli
    doc = json.loads(python)
    validate(doc)
    assert doc["horizoneval_result"] == RESULT_VERSION and doc["choice"]["data"]["model"] == "SeasonalNaive"


def test_the_mcp_server_returns_the_same_document(files):
    pytest.importorskip("mcp")
    import anyio
    from mcp import Client

    from horizoneval.mcp_server import build_server

    async def run():
        async with Client(build_server()) as client:
            listed = await client.list_tools()
            names = {t.name for t in listed.tools}
            res = await client.call_tool("score", _args(files))
            bad = await client.call_tool("score", {"forecasts": "/no/such/file.csv"})
            return names, res, bad

    names, res, bad = anyio.run(run)
    assert names == set(tools.TOOLS)
    assert res.content[0].text == tools.call_json("score", _args(files))
    assert res.structured_content == tools.call("score", _args(files))
    assert bad.is_error and "No such file" in bad.content[0].text


def test_check_and_diff_paths_agree(files, tmp_path):
    from horizoneval.checks import preflight

    assert tools.call("check", {"forecasts": str(files / "cv.csv")})["checks"] == json.loads(
        dumps(preflight(read(files / "cv.csv"))))
    with pytest.raises(SystemExit) as code:  # check exits with 1 on an error, 0 otherwise
        _cli(["check", str(files / "cv.csv"), "--json"])
    assert code.value.code == 0
    tools.call("score", _args(files, out=str(tmp_path / "a")))
    tools.call("score", _args(files, out=str(tmp_path / "b"), smape_zero="skip"))
    d = tools.call("diff", {"run_a": str(tmp_path / "a"), "run_b": str(tmp_path / "b")})
    assert d["causes"][0]["setting"] == "smape_zero"
    assert _cli(["diff", str(tmp_path / "a"), str(tmp_path / "b"), "--json"]) == dumps(d)
    assert tools.call("read_result", {"run": str(tmp_path / "a")}) == tools.call("score", _args(files))


def test_function_calling_definitions_are_valid_json_schema():
    import jsonschema

    for style in ["anthropic", "openai"]:
        defs = tools.definitions(style)
        assert len(defs) == len(tools.TOOLS)
    for t in tools.TOOLS.values():
        jsonschema.Draft202012Validator.check_schema(t.input_schema)
