#%%
import gymnasium as gym
import numpy as np

# Crear el ambiente de Blackjack
# Con natural=False se evita el bono extra por blackjack natural
env = gym.make("Blackjack-v1", render_mode='human')

# Parámetros del Q-learning
alpha = 0.1     # tasa de aprendizaje
gamma = 1.0     # factor de descuento
epsilon = 0.1   # probabilidad de exploración (epsilon-greedy)
num_episodes = 500000  # número de episodios para entrenar

# Inicializar la tabla Q como un diccionario:
# La clave es el estado (una tupla: (suma_jugador, carta_dealer, usable_ace))
# y el valor es un vector de tamaño 2 (una entrada para cada acción)
Q = {}

def get_Q(state):
    """
    Devuelve el vector Q para un estado dado.
    Si el estado no existe, lo inicializa con ceros.
    """
    if state not in Q:
        Q[state] = np.zeros(env.action_space.n)
    return Q[state]

def epsilon_greedy(state, epsilon):
    """
    Política epsilon-greedy: con probabilidad epsilon se elige una acción aleatoria,
    y con 1-epsilon se elige la acción con mayor valor Q.
    """
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(get_Q(state))

# Entrenamiento con Q-learning
for episode in range(num_episodes):
    # Reinicia el ambiente; env.reset() devuelve el estado y un diccionario de info
    state, info = env.reset()
    done = False

    while not done:
        # Seleccionar acción según la política epsilon-greedy
        action = epsilon_greedy(state, epsilon)
        
        # Ejecutar la acción en el ambiente
        next_state, reward, done, truncated, info = env.step(action)
        
        # Actualización Q-learning:
        # Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') − Q(s, a)]
        current_Q = get_Q(state)[action]
        if not done:
            target = reward + gamma * np.max(get_Q(next_state))
        else:
            target = reward  # Si se terminó el episodio, el target es simplemente el reward
        
        Q[state][action] = current_Q + alpha * (target - current_Q)
        
        # Mover al siguiente estado
        state = next_state

# Mensaje indicando que el entrenamiento ha finalizado
print("Entrenamiento completado.")

# Evaluación de la política aprendida (acción greedy: sin exploración)
num_eval = 10000
wins = 0

for _ in range(num_eval):
    state, info = env.reset()
    done = False
    while not done:
        # Elegimos siempre la acción con mayor valor Q
        action = np.argmax(get_Q(state))
        state, reward, done, truncated, info = env.step(action)
    if reward > 0:
        wins += 1

print("Tasa de victorias en evaluación:", wins / num_eval)
