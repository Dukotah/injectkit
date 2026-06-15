"""Tests for the ``injectkit gui`` subcommand.

The GUI command is point-and-shoot: ``injectkit gui`` should launch the local
web UI. These tests verify the subcommand parses and that ``_cmd_gui`` calls
``injectkit.web.serve`` with the parsed host/port/open_browser — without ever
starting a real server (``web.serve`` is mocked).
"""

from __future__ import annotations

from unittest import mock

from injectkit import cli
from injectkit.cli import EXIT_OK, build_parser


def test_gui_is_a_valid_subcommand():
    """`gui` parses and routes to the gui command with its defaults."""
    parser = build_parser()
    args = parser.parse_args(["gui"])
    assert args.command == "gui"
    assert args.host == "127.0.0.1"
    assert args.port == 8765
    assert args.no_open is False


def test_gui_subcommand_listed_in_metavar():
    """The subparser metavar advertises gui alongside scan/list/init."""
    parser = build_parser()
    help_text = parser.format_help()
    assert "gui" in help_text


def test_cmd_gui_calls_web_serve_with_defaults():
    """`_cmd_gui` forwards host/port and open_browser=True by default."""
    parser = build_parser()
    args = parser.parse_args(["gui"])
    with mock.patch("injectkit.web.serve") as serve:
        rc = cli._cmd_gui(args, out=None, err=None)
    assert rc == EXIT_OK
    serve.assert_called_once_with("127.0.0.1", 8765, open_browser=True)


def test_cmd_gui_honors_flags():
    """--host/--port/--no-open flow through to web.serve."""
    parser = build_parser()
    args = parser.parse_args(
        ["gui", "--host", "0.0.0.0", "--port", "9001", "--no-open"]
    )
    with mock.patch("injectkit.web.serve") as serve:
        rc = cli._cmd_gui(args, out=None, err=None)
    assert rc == EXIT_OK
    serve.assert_called_once_with("0.0.0.0", 9001, open_browser=False)


def test_main_dispatches_gui_without_starting_server():
    """main('gui') routes to the gui handler with the server mocked out."""
    with mock.patch("injectkit.web.serve") as serve:
        rc = cli.main(["gui", "--no-open"])
    assert rc == EXIT_OK
    serve.assert_called_once_with("127.0.0.1", 8765, open_browser=False)
