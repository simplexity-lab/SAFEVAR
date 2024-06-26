# World on Rail2.0


## Getting Started
* To run CARLA and train the models, make sure you are using a machine with **at least** a mid-end GPU.
* Please follow [INSTALL.md](https://github.com/dotchen/WorldOnRails/blob/release/docs/INSTALL.md) to setup the environment.


## Evaluation

**If you evaluating the pretrained weights, make sure you are launching CARLA with `-vulkan`!**

### Run
```bash
python evaluate_nocrash.py --town=Town01 --weather=train --resume
```
* Use defaults for _RAILS_.
* --OrigiCon=1 is use NSGA-II to search and default is 0 for original vehicle characteristics setting(OrigiCon=2 -> RS; OrigiCon=3 -> safeFuzzer).

### Pretrained weights
* [The model we used (NoCrash)](https://utexas.box.com/s/54m24gz5xwy1oagsqmgosch7pq561h2e)


## Acknowledgements
The `leaderboard` codes are built from the original [leaderboard](https://github.com/carla-simulator/leaderboard.git) repo.
The `scenariorunner` codes are from the original [scenario_runner](https://github.com/carla-simulator/scenario_runner.git) repo.
The `waypointer.py` GPS coordinate conversion codes are build from Marin Toromanoff's leadeboard submission.

