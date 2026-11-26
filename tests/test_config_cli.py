import json
from dataclasses import replace

import pytest
from typer.testing import CliRunner

from prunewise.cli import app
from prunewise.config import PRESETS, Config, load_config

runner = CliRunner()


def test_presets_and_toml_override(tmp_path):
    assert load_config("tests").n_rows == 40_000 and load_config("tests").row_group_sizes == (
        2000,
        8000,
    )
    assert load_config("demo").n_rows == 250_000
    path = tmp_path / "mine.toml"
    path.write_text('base = "tests"\nmax_files = 50\nrow_group_sizes = [500, 1000]\n')
    cfg = load_config(str(path))
    assert (cfg.max_files, cfg.row_group_sizes, cfg.n_rows, cfg.name) == (
        50,
        (500, 1000),
        40_000,
        "mine",
    )
    assert Config.from_dict(cfg.to_dict()) == cfg
    assert replace(PRESETS["tests"], max_files=1).max_files == 1


def test_bad_config_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown config"):
        load_config("nope")
    path = tmp_path / "bad.toml"
    path.write_text("bogus = 1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(str(path))


def run_ok(*args):
    result = runner.invoke(app, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result


def test_cli_pipeline_end_to_end_and_tamper_detection(tmp_path):
    d, out, lake = tmp_path / "data", tmp_path / "out", tmp_path / "lake"
    run_ok("generate", "--config", "tests", "--seed", 11, "--out", d)
    run_ok("analyze-workload", "--data", d, "--out", out / "workload.json")
    run_ok(
        "rank",
        "--data",
        d,
        "--workload",
        out / "workload.json",
        "--config",
        "tests",
        "--out",
        out / "ranking.json",
    )
    run_ok("materialize", "--data", d, "--ranking", out / "ranking.json", "--out", lake, "--top", 1)
    ok = run_ok(
        "validate",
        "--data",
        d,
        "--lake",
        lake,
        "--workload",
        out / "workload.json",
        "--ranking",
        out / "ranking.json",
        "--out",
        out / "validation.json",
    )
    assert "soundness violations 0" in ok.output.replace("\n", " ")
    report = run_ok(
        "report",
        "--ranking",
        out / "ranking.json",
        "--workload",
        out / "workload.json",
        "--validation",
        out / "validation.json",
        "--breakdown",
        1,
    )
    assert "## Ranked layouts" in report.output and "leading wildcard LIKE" in report.output
    assert "| soundness violations | 0 |" in report.output

    # overwrite the top layout's files with reversed row order: footers no longer match the plan
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    manifest = json.loads((lake / "manifest.json").read_text())
    victim = lake / manifest["layouts"][1]["layout_id"]
    for path in victim.rglob("part-0.parquet"):
        table = pq.read_table(path)
        pq.write_table(
            table.take(pc.sort_indices(table["sale_id"], sort_keys=[("sale_id", "descending")])),
            path,
            row_group_size=manifest["layouts"][1]["row_group_size"],
        )
    bad = runner.invoke(
        app,
        [
            "validate",
            "--data",
            str(d),
            "--lake",
            str(lake),
            "--workload",
            str(out / "workload.json"),
            "--ranking",
            str(out / "ranking.json"),
            "--out",
            str(out / "bad.json"),
        ],
    )
    assert bad.exit_code == 1
    assert json.loads((out / "bad.json").read_text())["summary"]["simulator_footer_mismatches"] > 0
