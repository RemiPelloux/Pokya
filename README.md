🃏 AI Poker Assistant
=====================

This project is an AI-powered poker assistant built to play and simulate Texas Hold'em using Reinforcement Learning (RL) and Monte Carlo strategies. The AI models are trained through a Genetic Algorithm that allows the AI players to evolve, improve their decision-making strategies, and potentially compete at a world champion level.

🏗️ Project Structure
---------------------

The project is structured into multiple components, which include the poker game mechanics, AI decision-making models, genetic algorithms for evolving the AI, and visualization tools for game analysis.

bash

Copy code

`.
├── ai
│   ├── ai_advisor.py        # Core AI advisor class with RL-based decision-making
│   ├── genetic_algorithm.py # Genetic algorithm for evolving the AI agents
├── data
│   ├── game_log.csv         # Stores game logs generated during the gameplay
├── models
│   ├── ai_advisor_X.pkl     # AI models for different players (stored after training)
├── src
│   ├── poker_game.py        # Main entry point to run poker games with AI agents
│   ├── visualization.py     # Visualization tools for game logs
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies`


🚀 Getting Started
------------------

### Prerequisites

To run this project, you'll need the following installed:

-   Python 3.9+
-   `torch` for building and training neural networks
-   `pandas` for handling game logs
-   `matplotlib` and `seaborn` for visualization

Install dependencies using:

bash

Copy code

`pip install -r requirements.txt`

### Running the Poker Game

You can start a game by running the following command:

bash

Copy code

`python src/poker_game.py`

The game runs for a set number of hands, with AI agents making decisions based on RL strategies. The game logs are saved in the `data/game_log.csv` file, which can be visualized later.

### AI Advisors

Each AI advisor can:

-   Use **Reinforcement Learning** (RL) to make decisions.
-   Be evolved using a **Genetic Algorithm** for improvement over generations.
-   Make **Monte Carlo** simulations to estimate winning probabilities during the game.

The AI models are trained and stored as `.pkl` files in the `models` folder, which can be reloaded between games.

### Handling Timeout for AI Decision

To avoid the AI getting stuck during decision-making, a **timeout mechanism** is used. If an AI takes too long to make a move, it defaults to **folding**. This is achieved using Python's `concurrent.futures`.

### Visualization

To visualize the performance of the AI agents and the game results, you can use the `visualization.py` script. This generates plots like player chip counts, pot sizes, and more from the `game_log.csv` file:

bash

Copy code

`python src/visualization.py`

🧠 AI Strategy
--------------

### Key Components

-   **Reinforcement Learning (RL)**: The AI uses an epsilon-greedy RL algorithm to balance between exploration and exploitation during gameplay.
-   **Monte Carlo Simulation**: AI advisors simulate outcomes to estimate the probabilities of winning with their current hand.
-   **Genetic Algorithm**: AI players evolve over generations to improve their strategies. The fittest advisors are selected based on their performance in each game.

### Advanced Strategies

-   **Bluffing**: AI agents can simulate bluffing strategies, attempting to deceive opponents based on the strength of their hand and game situation.
-   **Leaderboard of Starting Hands**: AI maintains a ranking of starting hands, which influences their pre-flop behavior.
-   **Granular Rewards**: AI agents are rewarded not only for winning hands but also for intermediate actions like progressing to the next betting round with a strong hand.

⚠️ Known Issues
---------------

-   **Raise Logic**: Ensure that raises are within the player's available chips, or the player will automatically go all-in.
-   **Action Validation**: Be mindful of invalid actions caused by exceeding the raise limit or attempting illegal moves (like calling when the player is in an `IN` state).

📊 Logs and Model Loading
-------------------------

-   Logs of epsilon values and game data are stored in `data/game_log.csv` and `epsilon_log.csv`.
-   AI models are loaded and saved from `models/ai_advisor_X.pkl`. If a model file doesn't exist, a new model is initialized.

💻 Contributing
---------------

Feel free to contribute by forking this repository and submitting pull requests. If you encounter any bugs or have feature requests, open an issue in the GitHub repository.

📄 License
----------

This project is licensed under the MIT License. See the LICENSE file for details.