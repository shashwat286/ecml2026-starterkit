import os

_HERE = os.path.dirname(os.path.abspath(__file__))

class Config:
    # --- ECML 2026 competition training (used by Training.py) ---
    # One fixed 120x150 map. Every episode samples a scene (which stations are
    # used), a line length (start + intermediate stops + target) and a train count.
    scenario_path = os.path.join(_HERE, "competition", "level_0_scenario_1.pkl")
    scenes        = ["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"]
    line_lengths  = [2, 3, 4, 5]   # competition goes up to ~8; long lines make resets slow
    min_agents    = 8
    max_agents    = 64             # also the padding size of the network batch; competition goes up to 532
    # Building a new schedule (line generator) takes 5-100 s on this map, so each
    # schedule is replayed this many times (with a new malfunction seed) before a new one is made.
    schedule_reuse = 4
    # Episode length comes from the timetable generator (~600-2000 steps).
    # Stop at intermediate stops? Off: in Flatland 4.2.6 a deliberate stop is charged
    # the 250 collision penalty, while a skipped stop costs only 50 (see waypoints.py).
    serve_intermediate_stops = False

    # --- Legacy random-map settings (only used by the old scripts) ---
    width  = 50
    height = 50
    n_agents = 12

    # Rail Generation Parameters (Sparse)
    n_cities = 5
    max_rails_between_cities = 3
    max_rail_pairs_in_city   = 2   # 2 pairs = 4 rails per city (max_rails_in_city=4)
    grid_mode = False
    max_episode_steps = 1500
    seed = 1
    
    # Malfunction Parameters
    malfunction_rate         = 0.01
    min_malfunction_duration = 20
    max_malfunction_duration = 50
    
    # Reward Logic
    # NOTE: only used by the legacy train_marl.py. Training.py uses Flatland's
    # built-in rewards, so changing these has no effect there.
    step_penalty = -0.3
    global_reward = 20.0
    
    # Observation and Action Space
    obs_size = 30   # NewAgentInitObs size
    action_size = 5 # (0: stop, 1: left, 2: straight, 3: right, 4: no-op/logic)
    
    # Training Parameters
    # Google Drive on Colab, local ./checkpoints everywhere else
    checkpoint_dir = ("/content/drive/MyDrive/Flatland/checkpoints"
                      if os.path.isdir("/content/drive")
                      else os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"))
    lr = 1e-4
    gamma = 0.99
    batch_size = 64
    alpha = 0.2 # Temperature parameter for SAC (if still used elsewhere)
    memory_size = 1000000 # Standard MARL buffer capacity
    
    # PPO Parameters
    ppo_epochs = 4
    ppo_clip = 0.2
    entropy_coef = 0.01
    
    # Coordination / Expert Logic Toggle
    use_expert = True
