"""Run the DQN experiment from Gym-DQN.ipynb on CartPole-v1.

Reuses the notebook's own code cells (imports, running_mean, DiscreteActionWrapper,
QFunction, DQNAgent) unchanged and runs one target-network variant per call.

Usage (from 6_gym-DQN/):
    PYTHONPATH=. python cartpole_dqn.py <no_target_network|update_every_call|update_every_20_calls>
        [--episodes 600] [--outdir plots/cartpole] [--seed 0]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))


def load_notebook_code(nb_path):
    """Exec the notebook cells that define imports, helpers and the DQN classes."""
    nb = json.load(open(nb_path))
    markers = ["import memory as mem", "def running_mean", "class DiscreteActionWrapper",
               "class QFunction", "class DQNAgent"]
    ns = {}
    for marker in markers:
        cells = [c for c in nb["cells"]
                 if c["cell_type"] == "code" and marker in "".join(c["source"])]
        assert len(cells) == 1, f"expected exactly one cell containing {marker!r}"
        src = "".join(cells[0]["source"])
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("%"))
        exec(src, ns)
    return ns


def plot_Q_function_generic(q_function, input_dims, plot_dim1, plot_dim2,
                            label_dim1, label_dim2, ranges, n=50):
    """3D surface of max_a Q(s,a) over two input dims; all other inputs are 0."""
    import numpy as np
    import pylab as plt
    from matplotlib import cm
    plt.rcParams.update({"font.size": 12})
    xxs = np.linspace(*ranges[plot_dim1], n)
    yys = np.linspace(*ranges[plot_dim2], n)
    XX, YY = np.meshgrid(xxs, yys)
    dots = np.zeros((XX.size, input_dims))
    dots[:, plot_dim1] = XX.ravel()
    dots[:, plot_dim2] = YY.ravel()
    values = q_function.maxQ(dots).reshape(XX.shape)

    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(XX, YY, values, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.view_init(elev=30, azim=45, roll=0)
    ax.set_xlabel(label_dim1)
    ax.set_ylabel(label_dim2)
    ax.set_zlabel("value")
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=["no_target_network", "update_every_call",
                                       "update_every_20_calls"])
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--test-episodes", type=int, default=50)
    p.add_argument("--outdir", default=os.path.join(HERE, "plots", "cartpole"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", choices=["cartpole", "pendulum"], default="cartpole")
    args = p.parse_args()

    ns = load_notebook_code(os.path.join(HERE, "Gym-DQN.ipynb"))
    gym, spaces, np, torch, plt = ns["gym"], ns["spaces"], ns["np"], ns["torch"], ns["plt"]
    DQNAgent, DiscreteActionWrapper, running_mean = (
        ns["DQNAgent"], ns["DiscreteActionWrapper"], ns["running_mean"])

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make({"cartpole": "CartPole-v1", "pendulum": "Pendulum-v1"}[args.env])
    if isinstance(env.action_space, spaces.Box):
        env = DiscreteActionWrapper(env, 5)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    ac_space, o_space = env.action_space, env.observation_space

    interval = {"no_target_network": 1, "update_every_call": 1, "update_every_20_calls": 20}
    q_agent = DQNAgent(o_space, ac_space, discount=0.95, eps=0.2,
                       target_update_interval=interval[args.variant])
    if args.variant == "no_target_network":
        q_agent.target_network = q_agent.Q  # use the online weights for the targets

    # --- training (same loop as the notebook) ---
    stats, losses = [], []
    max_steps = 500
    for i in range(args.episodes):
        total_reward = 0
        ob, _info = env.reset()
        for t in range(max_steps):
            a = q_agent.act(ob)
            ob_new, reward, done, trunc, _info = env.step(a)
            total_reward += reward
            q_agent.store_transition((ob, a, reward, ob_new, done))
            ob = ob_new
            if done:  # like the notebook: only stop on termination
                break
        losses.extend(q_agent.train(32))
        stats.append([i, total_reward, t + 1])
        if (i - 1) % 20 == 0:
            print(f"[{args.variant}] {i}: Done after {t+1} steps. Reward: {total_reward}",
                  flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    tag = args.variant

    stats_np = np.asarray(stats)
    N = 10
    plt.figure()
    plt.plot(running_mean(stats_np[:, 1], N))
    plt.title(f"{env.spec.id} return (moving average over {N} episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.savefig(os.path.join(args.outdir, f"return_train_{tag}.pdf"))
    plt.close()

    losses_np = np.asarray(losses)
    N = 250
    plt.figure()
    plt.plot(running_mean(losses_np, N))
    plt.title(f"{env.spec.id} loss (moving average over {N} gradient steps)")
    plt.xlabel("Gradient step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(args.outdir, f"loss_train_{tag}.pdf"))
    plt.close()

    # --- greedy evaluation ---
    test_stats = []
    for i in range(args.test_episodes):
        total_reward = 0
        ob, _info = env.reset()
        for t in range(max_steps):
            a = q_agent.act(ob, eps=0.0)
            ob_new, reward, done, trunc, _info = env.step(a)
            total_reward += reward
            ob = ob_new
            if done:
                break
        test_stats.append([i, total_reward, t + 1])
    test_stats = np.array(test_stats)
    mean, std = float(np.mean(test_stats[:, 1])), float(np.std(test_stats[:, 1]))
    print(f"[{tag}] Test return mean: {mean:.2f}  std: {std:.2f}", flush=True)

    # --- value function slices (other inputs = 0) ---
    ranges = {0: (-2.4, 2.4), 1: (-3, 3), 2: (-0.21, 0.21), 3: (-3, 3)}
    slices = [(0, 2, "Cart Pos", "Pole Angle", "cartpos_poleangle"),
              (0, 1, "Cart Pos", "Cart Vel", "cartpos_cartvel"),
              (2, 3, "Pole Angle", "Pole Vel", "poleangle_polevel")]
    for d1, d2, l1, l2, name in (slices if args.env == "cartpole" else []):
        fig = plot_Q_function_generic(q_agent.Q, o_space.shape[0], d1, d2, l1, l2, ranges)
        fig.savefig(os.path.join(args.outdir, f"value_function_{tag}_{name}.pdf"),
                    bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(args.outdir, f"results_{tag}.json"), "w") as f:
        json.dump({"variant": tag, "seed": args.seed, "train_episodes": args.episodes,
                   "test_episodes": args.test_episodes, "test_return_mean": mean,
                   "test_return_std": std,
                   "test_returns": test_stats[:, 1].tolist(),
                   "train_returns": stats_np[:, 1].tolist()}, f, indent=1)


if __name__ == "__main__":
    main()
