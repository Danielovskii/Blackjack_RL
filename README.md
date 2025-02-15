# Blackjack

## Agente de Blackjack con DQLearning

Este repositorio contiene el código de un agente de aprendizaje por refuerzo que aprende a jugar Blackjack mediante Q-Learning. El agente se entrena utilizando un diccionario de valores Q y, además, se visualiza la política y los valores de los estados mediante gráficos interactivos. También se incluyen funciones para ejecutar partidas de forma interactiva y evaluar el desempeño del agente.

---

## Contenido

- [Descripción](#descripción)
- [Dependencias e Instalación](#dependencias-e-instalación)
- [Uso](#uso)
  - [Entrenamiento del Agente](#entrenamiento-del-agente)
  - [Visualización de la Política](#visualización-de-la-política)
  - [Juego Interactivo](#juego-interactivo)
- [Implementación](#implementación)
  - [Componentes Clave del Agente](#componentes-claves-del-agente)
- [Resultados](#resultados)
  - [Política Aprendida y Gráficos](#política-aprendida)
  - [Interpretación de Gráficos](#interpretación-de-gráficos)
  - [Rendimiento Típico](#rendimiento-típico)
- [Licencia](#licencia)

---

## Descripción

El código utiliza el entorno `Blackjack-v1` de Gymnasium (con las reglas SAB habilitadas) para entrenar un agente que aprende a jugar Blackjack. El agente actualiza sus estimaciones de la función de valor de cada acción en cada estado mediante Q-Learning, almacenando los valores en un diccionario. Además, se generan gráficos 3D y mapas de calor que permiten visualizar tanto el valor esperado de cada estado como la acción óptima (política) aprendida, diferenciando entre estados en los que se posee un as utilizable y aquellos en los que no.

---

## Dependencias e Instalación

Para ejecutar este proyecto, asegúrate de tener instaladas las siguientes librerías de Python:

- [gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [numpy](https://numpy.org/)
- [tqdm](https://tqdm.github.io/)
- [termcolor](https://pypi.org/project/termcolor/)

Puedes instalarlas utilizando pip. Por ejemplo:

```bash
pip install gymnasium matplotlib seaborn numpy tqdm termcolor
```
---

## Uso

### Entrenamiento del Agente

```python
n_episodes = 500_000
learning_rate = 0.1
epsilon_decay = start_epsilon/(n_episodes/2)

agent = BlackjackAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon
)
```

### Visualización de la Política

```python
# Política con As usable
value_grid, policy_grid = create_grids(agent, usable_ace=True)
create_plots(value_grid, policy_grid, title="With usable ace")

# Política sin As usable
value_grid, policy_grid = create_grids(agent, usable_ace=False)
create_plots(value_grid, policy_grid, title="Without usable ace")
```

### Juego Interactivo

```python
# Jugar 3 partidas con renderizado
play_blackjack(agent, episodes=3, render_delay=1.0)
```
Durante cada partida se muestra la suma del jugador, la carta visible del dealer, si el as es utilizable y la acción tomada. Al final de cada partida, se imprimen los resultados y se actualizan las estadísticas.

---

## Implementación

### Componentes Claves del Agente

1. **Q-Table**:
  - Diccionario de estados -> valores de acciones
  - q_values[ (player_sum, dealer_card, usable_ace) ] = [Q(stick), Q(hit)]

2. Exploración epsilon-greedy:
  - **Epsilon inicial**: 1.0 (100% exploración)
  - **Epsilon final**: 0.1 (10% exploración)
  - Decaimiento lineal durante el entrenamiento

3. Actualización Q-Learning:

```python
future_q = (1 - terminated) * max(q_values[next_state])
td_error = reward + gamma * future_q - q_values[state][action]
q_values[state][action] += lr * td_error
```

---

## Resultados

### Política Aprendida

![Policy_Without_Usable_Ace](https://github.com/user-attachments/assets/4d7318ac-ad71-4b3f-80c4-1597dbf12948)

![Policy_With_Usable_Ace](https://github.com/user-attachments/assets/6504f7b1-323d-4101-adf4-1bc8d3417ee3)


### Interpretación de Gráficos
  - Superficie 3D: Valor esperado de cada estado
  - Mapa de calor: Acción óptima (Verde=Hit, Gris=Stick)

### Rendimiento Típico (500k episodios)
Win rate: 40.90%
Loss rate: 50.54%
Draw rate: 8.56%

---

## Licencia

Este proyecto es de uso público y se puede utilizar y modificar libremente para fines educativos y de investigación.

**Nota**: Basado en el ejemplo de Blackjack de Gymnasium y el libro "Reinforcement Learning: An Introduction" de Sutton y Barto.

