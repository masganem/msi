from machinations.gym_env import MachinationsEnv
from monopoly import *
import sys
import subprocess
import pathlib
import pickle
import os
import signal

env = MachinationsEnv(m, max_steps=200, record_history=True)

obs, info = env.reset()

steps = 10
if len(sys.argv) > 1:
    try:
        steps = int(sys.argv[1])
    except ValueError:
        print(f"[warn] Invalid episode count '{sys.argv[1]}', using 10.")

# Find indices for resources and connections we want to track
money_idx = next(i for i, r in enumerate(m.resources) if r.name == "Money")
estate_idx = next(i for i, r in enumerate(m.resources) if r.name == "Estate")
luck_idx = next(i for i, r in enumerate(m.resources) if r.name == "Luck")
player_idx = player.id
world_idx = world.id
get_rent_odds_idx = get_rent_odds.id

# Find indices in the arrays
rent_conn_idx = next(i for i, conn in enumerate(m.resource_connections) if conn.id == player_gets_rent.id)
get_rent_trigger_idx = next(i for i, trig in enumerate(m.triggers) if trig.id == get_rent.id)
pay_rent_conn_idx = next(i for i, conn in enumerate(m.resource_connections) if conn.id == player_pays_rent.id)
pay_rent_trigger_idx = next(i for i, trig in enumerate(m.triggers) if trig.id == pay_rent.id)
pass_go_conn_idx = next(i for i, conn in enumerate(m.resource_connections) if conn.id == player_gets_go.id)
pass_go_trigger_idx = next(i for i, trig in enumerate(m.triggers) if trig.id == pass_go.id)
random_rent_idx = random_rent_value1.id
pay_rent_odds_idx = pay_rent_odds.id
pass_go_odds_idx = pass_go_odds.id

for i in range(steps):
    action = env.action_space.sample()
    print(f"\nStep {i} ===========================")
    print(f"Player state before:")
    print(f"  Money: {env._model.X[player_idx, money_idx]}")
    print(f"  Estate: {env._model.X[player_idx, estate_idx]}")
    print(f"World state before:")
    print(f"  Money: {env._model.X[world_idx, money_idx]}")
    print(f"\nTrigger states before:")
    print(f"  Get rent odds (luck value): {env._model.X[get_rent_odds_idx, luck_idx]}")
    print(f"  Estate modifier on odds: -{env._model.X[player_idx, estate_idx]}")  # -1 per estate
    print(f"  Pay rent odds (luck value): {env._model.X[pay_rent_odds_idx, luck_idx]}")
    print(f"  Random rent value: {env._model.X[random_rent_idx, luck_idx]}")
    print(f"  Pass go odds (luck value): {env._model.X[pass_go_odds_idx, luck_idx]}")
    
    print(f"\nDebug modifier info:")
    print(f"  player_gets_rent.id: {player_gets_rent.id}")
    print(f"  E_R[3, 0] (connection id at index 3): {env._model.E_R[3, 0]}")
    print(f"  E_M array (modifiers):")
    for i in range(env._model.E_M.shape[0]):
        print(f"    E_M[{i}]: {env._model.E_M[i]}")
    
    print(f"\nActual T_e values BEFORE step:")
    print(f"  Get Rent T_e[3]: {env._model.T_e[3]}")  # player_gets_rent is index 3
    print(f"  Pay Rent T_e[1]: {env._model.T_e[1]}")  # player_pays_rent is index 1
    print(f"  Pass Go T_e[0]: {env._model.T_e[0]}")   # player_gets_go is index 0
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nActual T_e values AFTER step:")
    print(f"  Get Rent T_e[3]: {env._model.T_e[3]}")  # player_gets_rent is index 3
    print(f"  Pay Rent T_e[1]: {env._model.T_e[1]}")  # player_pays_rent is index 1
    print(f"  Pass Go T_e[0]: {env._model.T_e[0]}")   # player_gets_go is index 0
    
    # Get the mid-phase snapshot from renderer which shows the actual transfer rates
    renderer = info["renderer"]
    if renderer and renderer.history:
        mid_snap = next((snap for snap in renderer.history[-3:] if snap.get('phase') == 'mid'), None)
        if mid_snap:
            print(f"\nMoney transfers during step:")
            print(f"Get Rent (E_{player_gets_rent.id}):")
            print(f"  Transfer rate: {mid_snap['T_e'][3]}")  # player_gets_rent is index 3
            print(f"  Active: {mid_snap['E_R_active'][3]}")
            print(f"  Triggered: {mid_snap['V_active'][world_idx]}")
            print(f"  Trigger active: {mid_snap['E_G_active'][get_rent_trigger_idx]}")
            print(f"  R_triggered: {env._model.R_triggered[3]}")
            
            print(f"\nPay Rent (E_{player_pays_rent.id}):")
            print(f"  Transfer rate: {mid_snap['T_e'][1]}")  # player_pays_rent is index 1
            print(f"  Active: {mid_snap['E_R_active'][1]}")
            print(f"  Trigger active: {mid_snap['E_G_active'][pay_rent_trigger_idx]}")
            
            print(f"\nPass Go (E_{player_gets_go.id}):")
            print(f"  Transfer rate: {mid_snap['T_e'][0]}")  # player_gets_go is index 0
            print(f"  Active: {mid_snap['E_R_active'][0]}")
            print(f"  Trigger active: {mid_snap['E_G_active'][pass_go_trigger_idx]}")
    
    print(f"\nAfter step:")
    print(f"Player state after:")
    print(f"  Money: {env._model.X[player_idx, money_idx]}")
    print(f"  Estate: {env._model.X[player_idx, estate_idx]}")
    print(f"World state after:")
    print(f"  Money: {env._model.X[world_idx, money_idx]}")
    print(f"Action: {action}")
    print(f"Reward: {reward}")
    
    if terminated:
        break

# ================ #
# BEGIN RENDERING  #
# ================ #

if len(sys.argv) > 2:
    # Retrieve renderer and persist it for Manim playback
    r = info["renderer"]
    _renderer_pkl = pathlib.Path("render") / "renderer.pkl"
    with _renderer_pkl.open("wb") as _fp:
        pickle.dump(r, _fp)
    # --- Video quality ---------------------------------------------------
    # Default quality = medium; can override with SECOND CLI arg (l/m/h)
    quality = "m"
    q = sys.argv[2].lower()
    if q in ("l", "m", "h"):
        quality = q
    else:
        print(f"[warn] Unknown quality '{sys.argv[2]}', using 'm'.")

    cmd = ["manim", f"-q{quality}", "render/scene.py", "MachinationsScene"]
    # Suppress Manim's stdout/stderr to keep the console quiet
    proc = subprocess.Popen(cmd, preexec_fn=os.setsid)

    try:
        proc.wait()
    except KeyboardInterrupt:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        sys.exit(1)

    if quality == "h":
        vlc = subprocess.Popen(["vlc", "media/videos/scene/1080p60/MachinationsScene.mp4"], preexec_fn=os.setsid)
    elif quality == "m":
        vlc = subprocess.Popen(["vlc", "media/videos/scene/720p30/MachinationsScene.mp4"], preexec_fn=os.setsid)
    else:
        vlc = subprocess.Popen(["vlc", "media/videos/scene/480p15/MachinationsScene.mp4"], preexec_fn=os.setsid)
    try:
        vlc.wait()
    except KeyboardInterrupt:
        os.killpg(os.getpgid(vlc.pid), signal.SIGINT)
        sys.exit(1)
