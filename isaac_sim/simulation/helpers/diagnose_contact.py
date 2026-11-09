from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=150)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import ability_isaaclab.tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

env_cfg = parse_env_cfg("Isaac-Ability-Hand-Grasp-Direct-v0", num_envs=args_cli.num_envs)
env = gym.make("Isaac-Ability-Hand-Grasp-Direct-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

agent_cfg = load_cfg_from_registry("Isaac-Ability-Hand-Grasp-Direct-v0", "rsl_rl_cfg_entry_point")
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

obs, _ = env.get_observations()
unwrapped = env.unwrapped

print(f"\nDEBUG: env.unwrapped type: {type(unwrapped)}")
print(f"DEBUG: has _get_fingers_in_contact: {hasattr(unwrapped, '_get_fingers_in_contact')}")
print(f"DEBUG: has index_contact sensor: {hasattr(unwrapped, 'index_contact')}")

# per-step count of fingers in contact, across all envs, over the rollout
contact_counts_over_time = []

with torch.inference_mode():
    for step in range(args_cli.num_steps):
        actions = policy(obs)
        obs, rewards, dones, extras = env.step(actions)

        fingers_in_contact = unwrapped._get_fingers_in_contact()
        num_in_contact = fingers_in_contact.sum(dim=-1).float()
        contact_counts_over_time.append(num_in_contact.mean().item())

        if step < 5:
            print(f"DEBUG step {step}: fingers_in_contact[0]={fingers_in_contact[0].tolist()}, "
                  f"mean_contact={num_in_contact.mean().item():.4f}, "
                  f"reward_mean={rewards.mean().item():.4f}")

contact_counts_over_time = torch.tensor(contact_counts_over_time)

print("\n--- Contact diagnostic ---")
print(f"Mean fingers in contact across rollout: {contact_counts_over_time.mean().item():.3f}")
print(f"Std of fingers in contact across rollout: {contact_counts_over_time.std().item():.3f}")

first_half = contact_counts_over_time[: args_cli.num_steps // 2]
second_half = contact_counts_over_time[args_cli.num_steps // 2 :]
print(f"Mean fingers in contact, first half of rollout: {first_half.mean().item():.3f}")
print(f"Mean fingers in contact, second half of rollout: {second_half.mean().item():.3f}")
print("(if second half is meaningfully higher than first half, contact is building up")
print(" over time within an episode, consistent with intentional closing.")
print(" If flat or noisy with no clear rise, contact looks incidental/constant.)")

# fraction of (env, step) pairs with 3+ fingers simultaneously in contact
with torch.inference_mode():
    obs, _ = env.get_observations()
    high_contact_steps = 0
    total_steps = 0
    for step in range(args_cli.num_steps):
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        fingers_in_contact = unwrapped._get_fingers_in_contact()
        num_in_contact = fingers_in_contact.sum(dim=-1)
        high_contact_steps += (num_in_contact >= 3).sum().item()
        total_steps += num_in_contact.numel()

print(f"\nFraction of (env,step) with 3+ fingers simultaneously in contact: {high_contact_steps / total_steps:.3f}")

env.close()
simulation_app.close()