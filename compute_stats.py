#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from hydra_zen import store, zen

import conf.experiment  # Must import the config to add all components to the store!
from launch_stats import launch_stats_computation

if __name__ == "__main__":
    "============ Hydra-Zen ============"
    store.add_to_hydra_store(
        overwrite_ok=True
    )  # Overwrite Hydra's default config to update it
    zen(
        launch_stats_computation,
    ).hydra_main(
        config_name="base_experiment",
        version_base="1.3",  # Hydra base version
    )
