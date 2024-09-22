import csv
import os
import pickle
import random
import time

import torch
from scipy.optimize import linprog
from texasholdem import TexasHoldEm, ActionType, Deck, PlayerState
from texasholdem.evaluator import evaluate
import numpy as np
from models.rl_agent import RLAgent
import torch.nn as nn
from enum import Enum


class LocalPlayerState(Enum):
    IN = 1
    TO_CALL = 2
    ALL_IN = 3
    FOLDED = 4


class ComplexNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class AIAdvisor:
    def __init__(
        self,
        game: TexasHoldEm,
        player_id: int,
        use_rl=False,
        epsilon=1.0,
        epsilon_decay=0.99,
        min_epsilon=0.1,
        bluffing_probability=0.15,
    ):
        self.rl_model = self.initialize_rl_model()
        self.game = game
        self.player_id = player_id
        self.use_rl = use_rl
        self.epsilon = epsilon  # Probability of exploration
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.min_epsilon = min_epsilon  # Minimum value for epsilon
        self.bluffing_probability = bluffing_probability

        if use_rl:
            self.model = ComplexNN(input_size=10, output_size=3)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.loss_fn = torch.nn.MSELoss()
        else:
            self.model = None

    def initialize_rl_model(self):
        """
        Initializes the reinforcement learning model (ComplexNN).
        """
        input_size = 10  # Assuming the game state input has 10 features
        output_size = 3  # Three possible actions (Fold, Call, Raise)
        return ComplexNN(input_size, output_size)

    def get_advice(self):
        available_moves = self.game.get_available_moves()

        # Choose between pre-flop and post-flop strategy
        if not self.game.board:
            action_type, action_params = self.preflop_strategy()
        else:
            action_type, action_params = self.postflop_strategy()

        # Ensure the action is valid
        if action_type not in available_moves.action_types:
            action_type = ActionType.FOLD
            action_params = {}

        return action_type, action_params

    def preflop_strategy(self):
        hand = self.game.get_hand(self.player_id)
        position = self.get_player_position()

        # Calculate hand strength using pre-flop hand ranking
        hand_strength = self.preflop_hand_strength(hand)
        premium_hands = hand_strength > 0.8
        medium_hands = 0.5 < hand_strength <= 0.8

        # Early position requires stronger hands
        if position < self.game.max_players // 2:  # Early position
            if premium_hands:
                return ActionType.RAISE, {"amount": self.game.big_blind * 4}
            else:
                return ActionType.FOLD, {}

        # Late position can play speculative hands
        elif position >= self.game.max_players // 2:  # Late position
            if premium_hands:
                return ActionType.RAISE, {"amount": self.game.big_blind * 4}
            elif medium_hands or random.random() < self.bluffing_probability:
                return ActionType.RAISE, {"amount": self.game.big_blind * 3}
            else:
                return ActionType.FOLD, {}

        # Default action: fold weak hands in early positions
        return ActionType.FOLD, {}

    def should_bluff(self, board):
        # Simple bluffing logic: bluff if the board is weak (e.g., low cards)
        if all(card.rank <= 7 for card in board):
            return True
        return False

    def postflop_strategy(self):
        hand = self.game.get_hand(self.player_id)
        board = self.game.board
        hand_strength = self.monte_carlo_simulation(hand, board)

        # If the hand is strong, raise aggressively
        if hand_strength > 0.7:
            return ActionType.RAISE, {"amount": self.game.big_blind * 4}

        # If the hand is moderately strong, call
        elif hand_strength > 0.5:
            return ActionType.CALL, {}

        # Bluff selectively if the hand is weak and the board is weak
        if self.should_bluff(board):
            return ActionType.RAISE, {"amount": self.game.big_blind * 3}

        return ActionType.FOLD, {}

    def assign_rewards(self, action_type, hand, outcome, pot_size):
        """
        Granular reward system to encourage better learning.
        """
        reward = 0
        if outcome == "win":
            reward += pot_size * 0.05  # Reward based on pot size
        if action_type == ActionType.FOLD and self.should_have_folded(hand):
            reward += 10  # Reward for folding weak hands
        if action_type == ActionType.RAISE and outcome == "bluff_win":
            reward += 20  # Reward for successful bluff
        if action_type == ActionType.RAISE and outcome == "loss":
            reward -= 10  # Penalty for unsuccessful bluff
        return reward

    def should_have_folded(self, hand):
        # Returns True if the hand was weak and should have been folded
        return self.preflop_hand_strength(hand) < 0.5

    def rl_action(self):
        state = self.get_game_state()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            action = random.randint(0, 2)  # Random action: 0 (fold), 1 (call), 2 (raise)
        else:
            action_probs = self.model(state_tensor)
            action = torch.argmax(action_probs, dim=1).item()

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # Map action to ActionType
        action_type = [ActionType.FOLD, ActionType.CALL, ActionType.RAISE][action]
        action_params = {}

        if action_type == ActionType.RAISE:
            min_raise = self.game.min_raise()
            max_possible_raise = self.game.players[self.player_id].chips
            raise_amount = max(self.game.big_blind * 4, min_raise)
            raise_amount = min(raise_amount, max_possible_raise)

            if raise_amount < min_raise:
                action_type = ActionType.CALL  # Default to CALL if raise is invalid
            else:
                action_params["amount"] = raise_amount

        return action_type, action_params

    def log_epsilon(self, hand_number):
        log_path = "data/epsilon_log.csv"
        os.makedirs("data", exist_ok=True)

        with open(log_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["hand_number", "epsilon"])
            writer.writerow([hand_number, self.epsilon])

    def train_rl_model(self, reward):
        state = self.get_game_state()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = self.model(state_tensor)
        action = torch.argmax(action_probs).item()

        # Calculate the target Q-value
        target = reward + 0.99 * torch.max(action_probs).item()

        # Update the Q-values
        expected_value = torch.zeros_like(action_probs)
        expected_value[action] = target

        # Calculate the loss
        loss = self.loss_fn(action_probs, expected_value)

        # Backpropagate the loss and update the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate_hand_strength(self, hand, board):
        # Using Monte Carlo simulations for post-flop hand strength
        if not board:
            return self.preflop_hand_strength(hand)
        return self.monte_carlo_simulation(hand, board)

    def monte_carlo_simulation(self, hand, board, simulations=100):
        wins = 0
        active_players = [
            player
            for player in self.game.players
            if player.state != LocalPlayerState.FOLDED
        ]

        num_active_players = len(active_players)

        for _ in range(simulations):
            game_copy = self.game.copy(shuffle=True)
            deck = game_copy.deck
            if deck is None or len(deck.cards) == 0:
                deck = Deck()
                deck.shuffle()

            # Draw unknown cards to complete the board
            unknown_cards = deck.draw(5 - len(board))
            final_board = board + unknown_cards

            hands = {player.player_id: deck.draw(2) for player in active_players}

            our_rank = evaluate(hand, final_board)
            opponent_ranks = [
                evaluate(hands[player.player_id], final_board)
                for player in active_players
                if player.player_id != self.player_id
            ]

            if all(our_rank < opponent_rank for opponent_rank in opponent_ranks):
                wins += 1
            elif any(our_rank == opponent_rank for opponent_rank in opponent_ranks):
                wins += 0.5  # Tied hands count as half a win

        return wins / simulations

    def get_game_state(self):
        """
        Gathers the current game state as input for the RL model.
        """
        hand = self.game.get_hand(self.player_id)
        board = self.game.board
        chips = self.game.players[self.player_id].chips
        pot_size = sum(pot.get_total_amount() for pot in self.game.pots)
        position = self.get_player_position()
        to_call = self.game.chips_to_call(self.player_id)

        # Normalize game state data for input vector
        state = [
            chips / 1000,
            pot_size / 1000,
            len(board) / 5,  # Normalize number of board cards (max is 5)
            position / self.game.max_players,
            to_call / 1000,
            self.evaluate_hand_strength(hand, board),
        ]

        # Padding to ensure the state has exactly 10 dimensions
        state += [0] * (10 - len(state))
        return tuple(state)

    def advanced_strategy(self):
        """
        Combines Nash Equilibrium, hand strength evaluation, and position factor for optimal decision making.
        """
        hand = self.game.get_hand(self.player_id)
        board = self.game.board
        position = self.get_player_position()

        hand_strength = self.evaluate_hand_strength(hand, board)
        nash_action = self.calculate_nash_equilibrium(hand, board)

        # Adjust action based on position
        position_factor = (self.game.max_players - position) / self.game.max_players
        adjusted_strength = hand_strength * position_factor

        if nash_action == ActionType.RAISE and adjusted_strength > 0.7:
            raise_amount = max(self.game.big_blind * 4, self.game.min_raise())
            return ActionType.RAISE, {"amount": raise_amount}
        elif nash_action == ActionType.CALL and adjusted_strength > 0.4:
            return ActionType.CALL, {}
        else:
            return ActionType.FOLD, {}

    def get_player_position(self):
        """
        Determines the player's position relative to the dealer (button).
        """
        position = (self.player_id - self.game.btn_loc) % self.game.max_players
        return position

    def calculate_nash_equilibrium(self, hand, board):
        """
        Adjusts strategies based on Nash equilibrium solutions.
        """
        payoff_matrix = np.array(
            [
                [0, -1, -1],  # FOLD
                [0.5, 0, -0.5],  # CALL
                [1, 0.5, 0],  # RAISE
            ]
        )

        num_strategies = payoff_matrix.shape[0]
        c = np.ones(num_strategies)
        A_eq = np.ones((1, num_strategies))
        b_eq = np.array([1])  # Sum of probabilities must be 1
        bounds = [(0, 1)] * num_strategies

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        if result.success:
            mixed_strategy = result.x / result.x.sum()  # Normalize probabilities
            action = np.random.choice(["FOLD", "CALL", "RAISE"], p=mixed_strategy)

            if action == "FOLD":
                return ActionType.FOLD, {}
            elif action == "CALL":
                return ActionType.CALL, {}
            else:
                min_raise = self.game.min_raise()
                max_possible_raise = self.game.players[self.player_id].chips
                raise_amount = max(self.game.big_blind * 4, min_raise)
                raise_amount = min(raise_amount, max_possible_raise)
                return ActionType.RAISE, {"amount": raise_amount}
        else:
            return ActionType.FOLD, {}

    def preflop_hand_strength(self, hand):
        """
        Determines the strength of the hand before any community cards are dealt.
        """
        strong_hands = [
            ("A", "A"),
            ("K", "K"),
            ("Q", "Q"),
            ("A", "K"),
            ("K", "Q"),
            ("A", "Q"),
            ("J", "J"),
            ("T", "T"),
        ]

        rank_chars = {
            0: "2",
            1: "3",
            2: "4",
            3: "5",
            4: "6",
            5: "7",
            6: "8",
            7: "9",
            8: "T",
            9: "J",
            10: "Q",
            11: "K",
            12: "A",
        }

        ranks = [rank_chars[card.rank] for card in hand]

        if (ranks[0], ranks[1]) in strong_hands or (ranks[1], ranks[0]) in strong_hands:
            return 0.9
        else:
            return 0.5

    def save_model(self, model_path):
        """
        Saves the RL model and optimizer state to a file.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)

    def load_model(self, model_path):
        """
        Loads the RL model and optimizer from a file.
        If no model file is found or loading fails, it initializes a new model.
        """
        if os.path.exists(model_path):
            try:
                model_data = torch.load(model_path)
                self.model.load_state_dict(model_data['model_state_dict'])
                self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
                self.model.train()  # Ensure model is in training mode
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}. Initializing a new model.")
                self.model = self.initialize_rl_model()
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
                self.loss_fn = torch.nn.MSELoss()
                self.model.train()
        else:
            print(f"Model file not found at {model_path}. Creating a new model.")
            self.model = self.initialize_rl_model()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.loss_fn = torch.nn.MSELoss()
            self.model.train()




