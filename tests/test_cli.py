"""Generation defaults and input validation at the public CLI."""

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from paper2manim.cli import cli


@pytest.mark.parametrize(
    "args",
    [["--input", "The Pythagorean theorem"], ["--arxiv", "1706.03762", "--section", "Background"]],
)
def test_default_generation_uses_full_pipeline(args, monkeypatch):
    graph = MagicMock()
    graph.invoke.side_effect = lambda state, **kwargs: state
    monkeypatch.setattr("paper2manim.graphs.generation.build_generation_graph", lambda: graph)
    result = CliRunner().invoke(cli, ["generate", *args])
    assert result.exit_code == 0, result.output
    state = graph.invoke.call_args.args[0]
    assert graph.invoke.call_args.kwargs["config"]["max_concurrency"] == 1
    assert state["skip_render"] is False
    assert state["max_retries"] == 2
    assert state["max_visual_revisions"] == 2
    assert state["vlm_enabled"] is True
    assert state["emb_enabled"] is True


def test_zero_retries_and_code_only(monkeypatch):
    graph = MagicMock()
    graph.invoke.side_effect = lambda state, **kwargs: state
    monkeypatch.setattr("paper2manim.graphs.generation.build_generation_graph", lambda: graph)
    result = CliRunner().invoke(
        cli, ["generate", "--input", "Section text", "--max-retries", "0", "--no-render"]
    )
    assert result.exit_code == 0, result.output
    state = graph.invoke.call_args.args[0]
    assert state["max_retries"] == 0
    assert state["emb_readonly"] is True


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--input", "x", "--arxiv", "1706.03762"],
        ["--input", ""],
        ["--input", "x", "--section", "Method"],
    ],
)
def test_invalid_inputs_fail_before_invocation(args):
    result = CliRunner().invoke(cli, ["generate", *args])
    assert result.exit_code != 0


def test_generation_failure_returns_nonzero(monkeypatch):
    graph = MagicMock()
    graph.invoke.return_value = {"fatal_error": "no successful scenes"}
    monkeypatch.setattr("paper2manim.graphs.generation.build_generation_graph", lambda: graph)
    result = CliRunner().invoke(cli, ["generate", "--input", "Section text"])
    assert result.exit_code == 1
    assert "no successful scenes" in result.output
