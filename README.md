# Blackjack

# Agente de Blackjack con DQLearning

Este repositorio contiene el código de un agente de aprendizaje por refuerzo que aprende a jugar Blackjack mediante Q-Learning. Se entrena al agente utilizando un diccionario de valores Q y se visualiza la política y los valores del estado con gráficos. Además, se incluye una función interactiva para ejecutar partidas en modo visual y otra para evaluar el desempeño del agente.

---

## Contenido

- [Descripción](#descripción)
- [Dependencias e Instalación](#dependencias-e-instalación)
- [Ejecución del Código](#ejecución-del-código)
- [Estructura y Funcionamiento del Código](#estructura-y-funcionamiento-del-código)
  - [Inicialización y Configuración del Entorno](#inicialización-y-configuración-del-entorno)
  - [Definición del Agente](#definición-del-agente)
  - [Entrenamiento del Agente](#entrenamiento-del-agente)
  - [Visualización de la Política y Valores de Estado](#visualización-de-la-política-y-valores-de-estado)
  - [Ejecutar y Evaluar Partidas](#ejecutar-y-evaluar-partidas)
- [Licencia](#licencia)

---

## Descripción

El código utiliza el entorno `Blackjack-v1` de Gymnasium (con las reglas SAB habilitadas) para entrenar un agente que aprende a jugar Blackjack. El agente utiliza Q-Learning para actualizar sus estimaciones de la función de valor de cada acción en cada estado, almacenando los valores en un diccionario. Además, se generan gráficos 3D y mapas de calor para visualizar el valor del estado y la política óptima aprendida, diferenciando entre estados con o sin un as utilizable.

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

--- 
