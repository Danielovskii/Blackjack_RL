from __future__ import annotations
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import time

env = gym.make("Blackjack-v1", sab=True, render_mode="rgb_array")


# Reset the environment to get the first observation
done = False
observation, info = env.reset()

# Sample a random action
action = env.action_space.sample()

# Execute the action in our environment
observation, reward, terminated, truncated, info = env.step(action)

# Create de Blackjack Agent
class BlackjackAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))
    
    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool]
    ):

        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

# Parameters to train agent
learning_rate = 0.1
n_episodes = 500000
start_epsilon = 1.0
epsilon_decay = start_epsilon/(n_episodes/2)
final_epsilon = 0.1

agent =  BlackjackAgent(
    learning_rate = learning_rate, 
    initial_epsilon = start_epsilon,
    epsilon_decay = epsilon_decay,
    final_epsilon = final_epsilon
)

env = gym.wrappers.RecordEpisodeStatistics(env)
for episode in tqdm(range(n_episodes)):
    done = False
    obs, info = env.reset()

    # Play one episdoe
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # Update if the environment is terminated
        done = terminated or truncated
        obs = next_obs
    agent.decay_epsilon()

# Visualizing the policy
def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig


# state values & policy with usable ace (ace counts as 11)
value_grid, policy_grid = create_grids(agent, usable_ace=True)
fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
plt.show()

# state values & policy with usable ace (ace counts as 1)
value_grid, policy_grid = create_grids(agent, usable_ace=False)
fig2 = create_plots(value_grid, policy_grid, title="Without usable ace")
plt.show()

def play_blackjack(agent, episodes=1, render_delay=1.0, policy="trained"):
    test_env = gym.make("Blackjack-v1", sab=True, render_mode="human")
    results = {"wins": 0, "losses": 0, "draws": 0}
    
    for episode in range(episodes):
        obs, info = test_env.reset()
        done = False
        episode_reward = 0
        
        print(f"\n=== Episodio {episode + 1} ===")
        
        while not done:
            test_env.render()
            player_sum, dealer_card, usable_ace = obs
            print(f"\nJugador: {player_sum} | Dealer: {dealer_card} | As útil: {usable_ace}")
            
            # Selección de acción
            action = agent.get_action(tuple(obs)) if policy == "trained" else test_env.action_space.sample()
            print(f"Acción: {'HIT' if action == 1 else 'STICK'}")
            
            # Paso del entorno
            next_obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            episode_reward = reward
            
            # Manejo del resultado final
            if done:
                print("\n--- RESULTADO FINAL ---")
                print(f"Suma final jugador: {next_obs[0]}")
                
                # Método confiable para obtener suma del dealer
                if action == 0:  # Solo cuando el jugador hace STICK
                    dealer_sum = test_env.unwrapped.dealer[0] + test_env.unwrapped.dealer[1]
                    print(f"Suma final dealer: {dealer_sum}")
                else:
                    print("Dealer no jugó (jugador se pasó de 21)")
                
                print(f"Recompensa: {reward}")
                
                # Actualizar estadísticas
                if reward == 1:
                    results["wins"] += 1
                    print(colored("¡Victoria!", "green"))
                elif reward == -1:
                    results["losses"] += 1
                    print(colored("Derrota", "red"))
                else:
                    results["draws"] += 1
                    print(colored("Empate", "yellow"))
            
            obs = next_obs
            time.sleep(render_delay)
        
        print("\n" + "="*40 + "\n")
    
    test_env.close()
    
    # Mostrar estadísticas finales
    print("\n=== Estadísticas ===")
    print(f"Victorias: {results['wins']}/{episodes}")
    print(f"Derrotas: {results['losses']}/{episodes}")
    print(f"Empates: {results['draws']}/{episodes}")
    return results

# Para usar colores en la salida (opcional)
from termcolor import colored

# Ejecutar 3 partidas con el agente entrenado
play_results = play_blackjack(agent, 
                             episodes=3, 
                             render_delay=1.5, 
                             policy="trained")

# Ejecutar 3 partidas con política aleatoria para comparar
play_results_random = play_blackjack(agent, 
                                    episodes=3, 
                                    render_delay=1.5, 
                                    policy="random")


def test_policy(agent, episodes=1000):
    env = gym.make("Blackjack-v1", sab=True)
    results = []
    
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(tuple(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        results.append(reward)
    
    wins = sum(r == 1 for r in results)
    losses = sum(r == -1 for r in results)
    draws = sum(r == 0 for r in results)
    
    print(f"Win rate: {wins/episodes:.2%}")
    print(f"Loss rate: {losses/episodes:.2%}")
    print(f"Draw rate: {draws/episodes:.2%}")
    
    return results

# Evaluar el agente con 100,000 partidas
test_results = test_policy(agent, episodes=100_000)

play_blackjack(agent, episodes=3, render_delay = 1.0, policy="trained")

test_policy(agent, episodes=100_000)
