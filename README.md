# Living Lab Digital Twin
Official repository for the implementation of the digital twin of the **Living Lab** at the [National Research Council](https://www.pd.cnr.it/) of Padua.

## 1. 📋 Introduction
Inspired by [CityLearn](https://github.com/citylearn-project/CityLearn) **LivingLab** is a modular and object oriented _Reinforcement Learning_ (**RL**) environment for intelligent building energy management for comfort maintenance.

LivingLab follows the modern [Farama Gymnasium API](https://gymnasium.farama.org/), making it plug-and-play with popular RL libraries like Stable Baselines3.

### 1.1 ✨ Key Features
* **Modular Representation**: the Heat Pump, Thermal Battery and Photovoltaic System are isolated components.
* **Gymnasium Compliant**: ready to use with `env.reset()` and `env.step()`.
* **Highly Customizable**: Easily override the simulation dataset, devices parameters, and building dynamics via environment configurations.

### 1.2 📂 Project Structure
```bash
LivingLab/
├── data/                       # Time-series datasets (weather, pricing, loads)
├── livinglab/                  # Main environment package
│   ├── __init__.py             # Registers the Gym environments
│   ├── base.py                 # Base classes definition
│   ├── envs/
│   │   └── livinglab_env.py    # Core Gymnasium environment loop
│   ├── components/             # Object-oriented physical assets
│   │   ├── device.py           # Devices physical implementation
│   │   ├── battery.py          # Energy Storage System dynamics
│   │   └── dynamics.py         # Building temperature dynamics
│   └── utils/
│       ├── data_loader.py      # CSV parsing and episode handling
│       ├── preprocessing.py    # Preprocessing functions for data normalzation
│       └── rewards.py          # Modular reward calculation
└── examples/                   # Example scripts and agent implementations
    └── init_env.py             # Random agent quickstart
```

### 1.3 ⚙️ Installation
We recommend using a [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) virtual environment.

1. Clone the repository: 
    ```bash
    git clone https://github.com/Isla-lab/LivingLab.git
    ```
2. Create the miniconda environment:
    ```bash
    cd LivingLab
    conda create -n living-lab --python==3.10.18 -y
    conda activate living-lab
    ```

## 2. 🚀 Quick Start
You can easily use the environment by importing it and and calling calling `.gym.make()`.
```python
import gymnasium as gym
import livinglab # Registers the environment with default configurations

# 1. Initialize the environment
env = gym.make("LivingLab-v0")

# 2. Reset to start the episode
obs, info = env.reset()

done = False
while not done:
    # Sample a random action (e.g., battery charge/discharge rates)
    action = env.action_space.sample()
    
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated

env.close()
```

To have more insights on the environment initialization and usage you can also check out and run the provided test script:
```bash
cd examples
python init_env.py
```

### 2.1 🏗️ Customizing the Environment
You can easily override the default environment settings by passing keyword arguments directly to `gym_make()`. Thi allows for rapid testing on custom configurations without altering the core code.
```python
env = gym.make(
    "LivingLab-v0", 
    episode_length=24 # <- daily episodes instead of the full dataset
)
```

## 3. 🔧 Future Work
The environment will be continuously updated with future work covering:
* easier custom environment definition and loading via `.json` configuration files;
* out-of-the-box compatibility with [Omnisafe](https://github.com/PKU-Alignment/omnisafe) for Safe RL;
* improved devices and building dynamics modelling. 