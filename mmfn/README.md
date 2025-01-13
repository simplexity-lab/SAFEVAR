# mmfn

## Getting Started

- To run CARLA and train the models, make sure you are using a machine with **at least** a mid-end GPU.
- Please follow [INSTALL.md](https://github.com/Kin-Zhang/mmfn/blob/main/README.md) to setup the environment.

## Evaluation
### Run

```
python run_steps/phase0.1_Physical.py --config-name=eval
```
### Pretrained weights
- [The model we used ](https://hkustconnect-my.sharepoint.com/:f:/g/personal/qzhangcb_connect_ust_hk/EkUVbfjq4idJrOy1zdIv9fEBoQr5caY2TBDxtYeJVJ0ZrQ?e=4xJsxb)

## Acknowledgements
This implementation is based on codes from several repositories. Thanks for these authors who kindly open-sourcing their work to the community. 
* [LBC](https://github.com/dotchen/LearningByCheating),
* [WorldOnRails](https://github.com/dotchen/WorldOnRails),
* [Transfuser](https://github.com/autonomousvision/transfuser),
* [carla-brid-view](https://github.com/deepsense-ai/carla-birdeye-view), 
* [pylot](https://github.com/erdos-project/pylot),
* [CARLA Leaderboard 1.0](https://github.com/carla-simulator/leaderboard), 
* [Scenario Runner 1.0](https://github.com/carla-simulator/scenario_runner)