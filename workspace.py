import numpy as np
import src

import pickle
from pathlib import Path
from time import time

import gpytorch
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from experiments.ablation.ablation_model import AblationModel

from matplotlib import pyplot as plt


def get_map(cfg):
    with np.load(f"data/arrays/{cfg.map.geo_coordinate}.npz") as data:
        map = data["arr_0"]
    print(f"Loaded map with shape {map.shape}.")
    return map


def get_sensor(cfg, map, rng):
    rate = cfg.sensor.sensing_rate
    noise_scale = cfg.sensor.noise_scale
    sensor = src.sensors.PointSensor(
        matrix=map,
        env_extent=cfg.map.env_extent,
        rate=rate,
        noise_scale=noise_scale,
        rng=rng,
    )
    print(f"Initialized sensor with rate {rate} and noise scale {noise_scale}.")
    return sensor

def get_scalers(cfg, x_init, y_init):
    x_scaler = src.scalers.MinMaxScaler()
    x_scaler.fit(x_init)
    y_scaler = src.scalers.StandardScaler()
    y_scaler.fit(y_init)
    return x_scaler, y_scaler


def get_kernel(cfg, x_init):
    if cfg.kernel.name == "rbf":
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        kernel.base_kernel.lengthscale = cfg.kernel.lengthscale
        kernel.outputscale = cfg.kernel.outputscale
    elif cfg.kernel.name == "ak":
        kernel = gpytorch.kernels.ScaleKernel(
            src.kernels.AttentiveKernel(
                dim_input=x_init.shape[1],
                dim_hidden=cfg.kernel.dim_hidden,
                dim_output=cfg.kernel.dim_output,
                min_lengthscale=cfg.kernel.min_lengthscale,
                max_lengthscale=cfg.kernel.max_lengthscale,
            )
        )
    else:
        raise ValueError(f"Unknown kernel: {cfg.kernel}")
    print(f"Initialized kernel {cfg.kernel.name}.")
    return kernel



def get_model(cfg, x_init, y_init, x_scaler, y_scaler, kernel):
    model = AblationModel(
        num_inducing=cfg.model.num_inducing,
        learn_inducing=cfg.model.learn_inducing,
        learn_variational=cfg.model.learn_variational,
        x_train=x_init,
        y_train=y_init,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        kernel=kernel,
        noise_variance=cfg.likelihood.noise_variance,
        batch_size=cfg.model.batch_size,
        jitter=cfg.model.jitter,
        use_online_elbo=cfg.model.use_online_elbo,
    )
    print(f"Initialized model {cfg.model.name}.")
    return model


def model_update(cfg, model, evaluator):
    print("Updating model...")
    start = time()
    model.update_inducing(name=cfg.model.name)
    if not cfg.model.learn_variational:
        if "ssgp" in cfg.model.name:
            model.update_variational("ssgp")
        else:
            model.update_variational()
    losses = model.optimize(num_steps=cfg.num_train_steps)
    end = time()
    evaluator.training_times.append(end - start)
    evaluator.losses.extend(losses)
def get_evaluator(cfg, sensor):
    evaluator = src.utils.Evaluator(sensor, cfg.map.task_extent, cfg.eval_grid)
    print(f"Initialized evaluator.")
    return evaluator

def evaluation(model, evaluator):
    print("Evaluating model...")
    start = time()
    mean, std = model.predict(evaluator.eval_inputs)
    end = time()
    evaluator.prediction_times.append(end - start)
    evaluator.compute_metrics(mean, std)
    print("RMSE",evaluator.root_mean_square_error())



@hydra.main(version_base=None, config_path="experiments/ablation/configs", config_name="main")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # experiment setup
    rng = src.utils.set_random_seed(cfg.seed)
    map = get_map(cfg)
    sensor = get_sensor(cfg, map, rng)
    X = np.load("results/trainingv2n35w107/trajectories/episode_001_trajectory.npy")
    #X = np.load("n17e073_poam.npy")
    #print(X)
    X = X-31
    #print(np.max(X), np.min(X))
    Y = sensor.sense(X)

    x_init = X[0:500,:]
    y_init = Y[0:500,:]
    x_new = X[500:,:]
    y_new = Y[500:,:]
    x_scaler, y_scaler = get_scalers(cfg, x_init, y_init)
    kernel = get_kernel(cfg, x_init)
    model = get_model(cfg, x_init, y_init, x_scaler, y_scaler, kernel)
    model.optimize(num_steps=1)
    evaluator = get_evaluator(cfg, sensor)
    evaluator.add_data(x_new, y_new)
    model.add_data(x_new, y_new)
    model_update(cfg, model, evaluator)
    evaluation(model, evaluator)

    model_update(cfg, model, evaluator)
    evaluation(model, evaluator)


if __name__ == "__main__":
    main()
