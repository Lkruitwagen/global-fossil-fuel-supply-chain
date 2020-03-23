"""Application entry point."""
from pathlib import Path
from typing import Dict

from kedro.context import KedroContext, load_context
from kedro.pipeline import Pipeline

from src.ffsc.pipeline import create_pipelines


class ProjectContext(KedroContext):
    """Users can override the remaining methods from the parent class here,
    or create new ones (e.g. as required by plugins)
    """

    project_name = "FFSC"
    project_version = "0.15.7"

    def _get_pipelines(self) -> Dict[str, Pipeline]:
        return create_pipelines()


def run_package():
    # entry point for running pip-install projects
    # using `<project_package>` command
    project_context = load_context(Path.cwd())
    project_context.run()


if __name__ == "__main__":
    # entry point for running pip-installed projects
    # using `python -m <project_package>.run` command
    run_package()
