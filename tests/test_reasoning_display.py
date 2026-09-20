import re

from nanochat.reasoning_display import ThinkRenderer


def test_every_stream_boundary_preserves_text() -> None:
    text = "Before <think>2+2=4\n</think>\n#### 4"
    for index in range(len(text) + 1):
        renderer = ThinkRenderer(True)
        result = renderer.feed(text[:index]) + renderer.feed(text[index:]) + renderer.finish()
        assert re.sub(r"\x1b\[[0-9;]*m", "", result) == text
        assert "\033[2m<think>" in result
        assert "</think>\033[0m" in result


def test_redirected_output_and_incomplete_tag() -> None:
    renderer = ThinkRenderer(False)
    assert renderer.feed("<think>x</think>") + renderer.finish() == "<think>x</think>"
    renderer = ThinkRenderer(True)
    assert renderer.feed("<thi") == ""
    assert renderer.finish() == "<thi"
    renderer = ThinkRenderer(True)
    assert renderer.feed("<think>x") + renderer.finish() == "\033[2m<think>x\033[0m"


def test_cli_sources_and_existing_eval_api_default() -> None:
    from inspect import signature

    from scripts.chat_cli import build_parser
    from scripts.chat_eval import run_chat_eval

    parser = build_parser()
    assert parser.parse_args([]).source == "sft"
    for source in ("sft", "rl", "reason_sft", "reason_rl"):
        args = parser.parse_args(["--source", source, "--max-new-tokens", "777"])
        assert args.source == source and args.max_new_tokens == 777
    assert signature(run_chat_eval).parameters["max_new_tokens"].default == 512
