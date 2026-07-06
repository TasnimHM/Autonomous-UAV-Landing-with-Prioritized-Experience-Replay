#!/usr/bin/env python3
"""
per_training.py — Offline PER training for expert gating policy.

Updated for scaled terminal reward design:
  - Reward clipping REMOVED — terminal rewards (+50/+35/+20/+10) must
    not be clipped or all landings look identical to the policy.
  - Step rewards are already small (-0.52 to +0.5), only terminals are large.
  - CQL penalty kept for offline stability.
  - Smaller gamma (0.95) appropriate for short landing episodes.
  - Soft target network update (tau) instead of hard copy.
  - Learning rate scheduler to reduce lr when loss plateaus.

Terminal reward design (from per_logger.py):
  Successful landing:
    error < 3m  → +50   excellent
    error < 7m  → +35   good
    error < 12m → +20   okay
    error > 12m → +10   poor but completed
  Timed out:
    error < 15m → +25   partial success
    error < 30m →   0   mediocre
    error > 30m → -10   failure

Usage:
    python3 per_training.py \\
        --data ../runs/per_data/per_transitions.csv \\
        --epochs 100 \\
        --batch_size 128 \\
        --lr 0.0003 \\
        --output_dir ../runs/per_data/results

Output:
    results/
      trained_policy.pt      (PyTorch model + normalization stats)
      training_log.csv       (loss and metrics per epoch)
      training_curves.png    (4-panel diagnostic plot)
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import namedtuple

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Transition ─────────────────────────────────────────────────────────────────
Transition = namedtuple('Transition',
    ['state', 'action', 'reward', 'next_state', 'done', 'is_hard'])

# ── Dimensions ────────────────────────────────────────────────────────────────
STATE_DIM  = 12
ACTION_DIM = 2

STATE_COLS = [
    's_cx_large', 's_cy_large', 's_w_large', 's_h_large',
    's_cx_small', 's_cy_small', 's_w_small', 's_h_small',
    's_error_x',  's_error_y',  's_size',    's_altitude'
]
NEXT_STATE_COLS = [
    'ns_cx_large', 'ns_cy_large', 'ns_w_large', 'ns_h_large',
    'ns_cx_small', 'ns_cy_small', 'ns_w_small', 'ns_h_small',
    'ns_error_x',  'ns_error_y',  'ns_size',    'ns_altitude'
]


# ══════════════════════════════════════════════════════════════════════════════
# PER Buffer
# ══════════════════════════════════════════════════════════════════════════════

class PERBuffer:
    """
    Prioritized Experience Replay buffer.
    NO reward clipping — terminal rewards (+50/+35/+20/+10) carry the
    accuracy signal and must not be crushed to +1.
    Hard transitions get hard_boost priority multiplier.
    """

    def __init__(self, alpha=0.6, beta=0.4, hard_boost=3.0):
        self.alpha      = alpha
        self.beta       = beta
        self.hard_boost = hard_boost

        self.transitions = []
        self.priorities  = []
        self.state_mean  = None
        self.state_std   = None

    def load_from_csv(self, csv_path: str):
        print(f"Loading transitions from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Remove no-detection transitions
        df = df[df['action'] != -1].reset_index(drop=True)
        print(f"  Total transitions:  {len(df)}")
        print(f"  Hard transitions:   {df['is_hard'].sum()} "
              f"({df['is_hard'].mean()*100:.1f}%)")
        print(f"  Action 0 (far):     {(df['action']==0).sum()}")
        print(f"  Action 1 (near):    {(df['action']==1).sum()}")
        print(f"  Terminal steps:     {df['done'].sum()}")

        # Reward distribution summary
        rewards_raw = df['reward'].values.astype(np.float32)
        print(f"\n  Reward stats (NO clipping):")
        print(f"    min={rewards_raw.min():.3f}  max={rewards_raw.max():.3f}"
              f"  mean={rewards_raw.mean():.4f}  std={rewards_raw.std():.4f}")
        print(f"    Terminal-like (>9): {(rewards_raw > 9).sum()} steps")

        # States
        states      = df[STATE_COLS].values.astype(np.float32)
        next_states = df[NEXT_STATE_COLS].values.astype(np.float32)

        # Normalize states
        self.state_mean = states.mean(axis=0)
        self.state_std  = states.std(axis=0) + 1e-8
        states_norm      = (states      - self.state_mean) / self.state_std
        next_states_norm = (next_states - self.state_mean) / self.state_std

        # NO reward clipping — preserve terminal signal
        rewards = rewards_raw

        for i in range(len(df)):
            row = df.iloc[i]
            t   = Transition(
                state      = states_norm[i],
                action     = int(row['action']),
                reward     = float(rewards[i]),
                next_state = next_states_norm[i],
                done       = bool(row['done']),
                is_hard    = bool(row['is_hard']),
            )
            self.transitions.append(t)
            priority = self.hard_boost if t.is_hard else 1.0
            self.priorities.append(priority)

        self.priorities = np.array(self.priorities, dtype=np.float32)
        print(f"\n  Buffer loaded: {len(self.transitions)} transitions")
        print(f"  Reward clipping: DISABLED (terminal signal preserved)\n")
        return self

    def sample(self, batch_size: int):
        probs   = self.priorities ** self.alpha
        probs   = probs / probs.sum()
        indices = np.random.choice(len(self.transitions), batch_size,
                                   p=probs, replace=False)
        weights = (len(self.transitions) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        batch       = [self.transitions[i] for i in indices]
        states      = torch.FloatTensor(np.array([t.state      for t in batch]))
        actions     = torch.LongTensor( np.array([t.action     for t in batch]))
        rewards     = torch.FloatTensor(np.array([t.reward     for t in batch]))
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
        dones       = torch.FloatTensor(np.array([t.done       for t in batch]))
        weights     = torch.FloatTensor(weights)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = abs(float(td_error)) + 1e-6
            if self.transitions[idx].is_hard:
                priority *= self.hard_boost
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.transitions)


# ══════════════════════════════════════════════════════════════════════════════
# MLP Policy
# ══════════════════════════════════════════════════════════════════════════════

class GatingPolicy(nn.Module):
    """
    MLP Q-network: 12 → 128 → 64 → 2
    Input:  12-dim normalized state vector
    Output: Q values for [far expert, near expert]
    """

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

    def select_action(self, state_np, state_mean, state_std):
        """Select action for a raw (unnormalized) state numpy array."""
        state_norm = (state_np - state_mean) / state_std
        state_t    = torch.FloatTensor(state_norm).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            q_values = self.forward(state_t)
        self.train()
        return int(q_values.argmax().item())


# ══════════════════════════════════════════════════════════════════════════════
# Soft target update
# ══════════════════════════════════════════════════════════════════════════════

def soft_update(source: nn.Module, target: nn.Module, tau: float):
    """Gradually blend source weights into target — more stable than hard copy."""
    for src_p, tgt_p in zip(source.parameters(), target.parameters()):
        tgt_p.data.copy_(tau * src_p.data + (1.0 - tau) * tgt_p.data)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    print("\n" + "="*60)
    print("  PER Training — Expert Gating Policy")
    print("  Scaled terminal rewards — NO reward clipping")
    print("="*60)

    # ── Load buffer ────────────────────────────────────────────────────────────
    buffer = PERBuffer(
        alpha      = args.alpha,
        beta       = args.beta,
        hard_boost = args.hard_boost,
    ).load_from_csv(args.data)

    if len(buffer) < args.batch_size:
        print(f"ERROR: buffer size {len(buffer)} < batch_size {args.batch_size}")
        return

    # ── Models ─────────────────────────────────────────────────────────────────
    policy = GatingPolicy()
    target = GatingPolicy()
    target.load_state_dict(policy.state_dict())
    target.eval()

    # ── Optimizer + scheduler ──────────────────────────────────────────────────
    optimizer = optim.Adam(policy.parameters(), lr=args.lr,
                           weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    criterion = nn.SmoothL1Loss(reduction='none')  # Huber loss

    # ── Log ────────────────────────────────────────────────────────────────────
    log = {
        'epoch': [], 'loss': [], 'mean_q': [],
        'mean_reward': [], 'lr': [], 'q_std': []
    }

    print(f"Training: {args.epochs} epochs | "
          f"batch={args.batch_size} | lr={args.lr} | gamma={args.gamma}")
    print(f"PER: alpha={args.alpha} | beta={args.beta} | "
          f"hard_boost={args.hard_boost}")
    print(f"CQL weight: {args.cql_weight} | tau: {args.tau}\n")

    steps_per_epoch = len(buffer) // args.batch_size

    for epoch in range(args.epochs):
        policy.train()
        epoch_loss = epoch_q = epoch_reward = epoch_q_std = 0.0

        for step in range(steps_per_epoch):

            states, actions, rewards, next_states, dones, weights, indices = \
                buffer.sample(args.batch_size)

            # Current Q values
            q_values = policy(states)
            q_taken  = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target Q values (Double DQN style)
            with torch.no_grad():
                next_q   = target(next_states).max(1)[0]
                q_target = rewards + args.gamma * next_q * (1.0 - dones)

            # Bellman loss weighted by PER importance sampling
            td_errors = (q_taken - q_target).detach().numpy()
            losses    = criterion(q_taken, q_target)
            bell_loss = (weights * losses).mean()

            # Conservative Q Learning penalty
            # Prevents overestimation of Q values in offline RL
            cql_loss = args.cql_weight * (
                torch.logsumexp(q_values, dim=1).mean() - q_taken.mean()
            )

            loss = bell_loss + cql_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            optimizer.step()

            # Soft target update
            soft_update(policy, target, args.tau)

            # Update PER priorities
            buffer.update_priorities(indices, td_errors)

            epoch_loss   += bell_loss.item()
            epoch_q      += q_taken.mean().item()
            epoch_reward += rewards.mean().item()
            epoch_q_std  += q_taken.std().item()

        # Epoch averages
        avg_loss   = epoch_loss   / steps_per_epoch
        avg_q      = epoch_q      / steps_per_epoch
        avg_reward = epoch_reward / steps_per_epoch
        avg_q_std  = epoch_q_std  / steps_per_epoch
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(avg_loss)

        # Anneal beta toward 1.0 over training
        buffer.beta = min(1.0, buffer.beta + (1.0 - args.beta) / args.epochs)

        log['epoch'].append(epoch + 1)
        log['loss'].append(avg_loss)
        log['mean_q'].append(avg_q)
        log['mean_reward'].append(avg_reward)
        log['lr'].append(current_lr)
        log['q_std'].append(avg_q_std)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Loss={avg_loss:.4f} | "
                  f"Q={avg_q:.4f} ± {avg_q_std:.4f} | "
                  f"Reward={avg_reward:.4f} | "
                  f"LR={current_lr:.6f} | "
                  f"Beta={buffer.beta:.3f}")

    # ── Save model ─────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'trained_policy.pt')
    torch.save({
        'model_state_dict': policy.state_dict(),
        'state_mean':       buffer.state_mean,
        'state_std':        buffer.state_std,
        'args':             vars(args),
    }, save_path)
    print(f"\n✅ Model saved → {save_path}")

    # ── Save log ───────────────────────────────────────────────────────────────
    log_path = os.path.join(args.output_dir, 'training_log.csv')
    pd.DataFrame(log).to_csv(log_path, index=False)
    print(f"✅ Training log → {log_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(log['epoch'], log['loss'], color='crimson')
    axes[0].set_title('Training Loss (Huber)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    axes[1].plot(log['epoch'], log['mean_q'], color='steelblue')
    axes[1].fill_between(
        log['epoch'],
        np.array(log['mean_q']) - np.array(log['q_std']),
        np.array(log['mean_q']) + np.array(log['q_std']),
        alpha=0.2, color='steelblue'
    )
    axes[1].set_title('Mean Q Value ± Std')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Q Value')
    axes[1].grid(True)

    axes[2].plot(log['epoch'], log['mean_reward'], color='seagreen')
    axes[2].set_title('Mean Reward per Epoch')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Reward')
    axes[2].grid(True)

    axes[3].plot(log['epoch'], log['lr'], color='darkorange')
    axes[3].set_title('Learning Rate Schedule')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('LR')
    axes[3].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✅ Training curves → {plot_path}")
    print("\n🎉 Training complete!")

    return policy, buffer


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PER Training — Expert Gating Policy (scaled terminal rewards)'
    )
    parser.add_argument('--data',       type=str,   default='per_transitions.csv')
    parser.add_argument('--output_dir', type=str,   default='.')
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=128)
    parser.add_argument('--lr',         type=float, default=0.0003)
    parser.add_argument('--gamma',      type=float, default=0.95,
                        help='Discount factor — 0.95 suits short episodes')
    parser.add_argument('--alpha',      type=float, default=0.6,
                        help='PER priority exponent')
    parser.add_argument('--beta',       type=float, default=0.4,
                        help='PER importance sampling (anneals to 1.0)')
    parser.add_argument('--hard_boost', type=float, default=3.0,
                        help='Priority multiplier for hard transitions')
    parser.add_argument('--cql_weight', type=float, default=0.1,
                        help='Conservative Q Learning penalty weight')
    parser.add_argument('--tau',        type=float, default=0.005,
                        help='Soft target network update rate')

    args = parser.parse_args()
    train(args)