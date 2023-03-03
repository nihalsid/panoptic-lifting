# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

from omegaconf import OmegaConf
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers.logger import DummyExperiment
from pytorch_lightning.loggers.logger import rank_zero_experiment


class FilesystemLogger(Logger):

    @property
    def version(self) -> Union[int, str]:
        return 0

    @property
    def name(self) -> str:
        return "fslogger"

    # noinspection PyMethodOverriding
    def log_hyperparams(self, params: argparse.Namespace):
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        pass

    def __init__(self, experiment_config, **_kwargs):
        super().__init__()
        self.experiment_config = experiment_config
        self._experiment = None
        # noinspection PyStatementEffect
        self.experiment

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is None:
            self._experiment = DummyExperiment()
            experiment_dir = Path("runs", self.experiment_config["experiment"])
            experiment_dir.mkdir(exist_ok=True, parents=True)

            src_folders = ['config', 'data/splits', 'model', 'tests', 'trainer', 'util', 'data_processing', 'dataset']
            sources = []
            for src in src_folders:
                sources.extend(list(Path(".").glob(f'{src}/**/*')))

            files_to_copy = [x for x in sources if x.suffix in [".py", ".pyx", ".txt", ".so", ".pyd", ".h", ".cu", ".c", '.cpp', ".html"] and x.parts[0] != "runs" and x.parts[0] != "wandb"]

            for f in files_to_copy:
                Path(experiment_dir, "code", f).parents[0].mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, Path(experiment_dir, "code", f))

            Path(experiment_dir, "config.yaml").write_text(OmegaConf.to_yaml(self.experiment_config))

        return self._experiment
