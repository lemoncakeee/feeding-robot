"""Shared test options and fixtures."""

from pathlib import Path

import pytest

from feeding_robot.simulation.sim import Scene


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", help="run tests against a real Ollama server")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="needs --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def scene() -> Scene:
    return Scene.load(Path(__file__).parent.parent / "scene.yaml")
