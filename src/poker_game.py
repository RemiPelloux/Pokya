import os
import pandas as pd
from texasholdem import TexasHoldEm, ActionType, PlayerState
from ai.ai_advisor import AIAdvisor
from ai.genetic_algorithm import genetic_algorithm


def safe_get_advice(ai_advisor, timeout=30):
    """
    Safely gets advice from AIAdvisor with a timeout.
    If the advisor takes too long, default to folding.
    """
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(ai_advisor.get_advice)
        try:
            action_type, action_params = future.result(timeout=timeout)
            return action_type, action_params
        except concurrent.futures.TimeoutError:
            print("AI took too long to make a decision. Defaulting to FOLD.")
            return ActionType.FOLD, {}


def main():
    # Initialize the game with 5 players
    game = TexasHoldEm(buyin=5000, big_blind=20, small_blind=10, max_players=5)

    # Run genetic algorithm to evolve AI advisors
    evolved_population = genetic_algorithm(
        game, population_size=11, generations=54, num_parents=5, mutation_rate=0.02
    )

    # Use the best evolved advisor for the game
    best_advisor = evolved_population[0]

    # Initialize AI Advisors for each player
    ai_advisors = [
        best_advisor,
        AIAdvisor(game, player_id=1, use_rl=True),
        AIAdvisor(game, player_id=2, use_rl=True),
        AIAdvisor(game, player_id=3, use_rl=False),
        AIAdvisor(game, player_id=4, use_rl=False),
    ]

    # Load models if they exist
    for i, advisor in enumerate(ai_advisors):
        model_path = f"models/ai_advisor_{i}.pkl"
        advisor.load_model(model_path)

    # Initialize data storage
    all_game_data = []  # Store all hands data

    # Run the game for X hands
    for hand_number in range(500):
        if hand_number % 50 == 0:
            print(f"Hand {hand_number + 1}")

        # Start the hand
        game.start_hand()
        hand_data = []  # Data for the current hand

        # Initialize a raise counter for the current hand
        raise_counter = {player_id: 0 for player_id in range(game.max_players)}

        # Inside the game loop
        while game.is_hand_running():
            current_player = game.current_player

            # Get action advice from AI
            try:
                action_type, action_params = safe_get_advice(ai_advisors[current_player])

                # Ensure action_params meet the game's rules
                if action_type == ActionType.RAISE:
                    min_raise = game.min_raise()
                    if action_params.get("amount", 0) < min_raise:
                        action_params["amount"] = min_raise

                    # Get player's available chips
                    available_chips = game.players[current_player].chips
                    raise_amount = action_params.get("amount", 0)

                    # If the raise is more than available chips, go all-in
                    if raise_amount > available_chips:
                        print(
                            f"Player {current_player} cannot raise {raise_amount}, going all-in with {available_chips}")
                        action_type = ActionType.ALL_IN
                        total = available_chips
                    else:
                        # Translate amount to total using value_to_total
                        total = game.value_to_total(raise_amount, current_player)

                    # Check for All-In conditions before taking action
                    if available_chips == total:
                        action_type = ActionType.ALL_IN

                    # Limit the number of raises per player
                    if raise_counter[current_player] >= 2:
                        # print(f"Player {current_player} has already raised twice, limiting to CALL or CHECK.")
                        # Check the player's state: if IN, allow CHECK, else CALL
                        if game.players[current_player].state == PlayerState.IN:
                            action_type = ActionType.CHECK
                            total = 0
                        else:
                            action_type = ActionType.CALL
                            total = game.chips_to_call(current_player)

                    raise_counter[current_player] += 1

                    # print(f"Player {current_player} decided to {action_type.name} with total: {total}")
                    game.take_action(action_type, total=total)

                else:
                    game.take_action(action_type)

            except ValueError as e:
                print(f"Invalid action: {e}")
                return 1
            except Exception as e:
                print(f"Unexpected error: {e}")
                return 1

        # Log data at the end of the hand
        total_pot_size = sum(pot.get_total_amount() for pot in game.pots)
        for player_id in range(game.max_players):
            hand_data.append(
                {
                    "hand_number": hand_number + 1,
                    "player_id": player_id,
                    "pot_size": total_pot_size,
                    "player_chips": game.players[player_id].chips,
                    "player_state": game.players[player_id].state.name,
                }
            )
        all_game_data.extend(hand_data)  # Add this hand's data to all_game_data

        # Reset player states for the next hand, except for players who are OUT
        for player in game.players:
            if player.state != PlayerState.OUT:
                player.state = PlayerState.IN

        # Shuffle the deck and reset the board for the next hand
        game.deck.shuffle()
        game.board = []  # Clear the board

    # Save all game data to CSV at the end with a date-time stamp in filename
    game_data_df = pd.DataFrame(all_game_data)
    os.makedirs("data", exist_ok=True)
    game_data_df.to_csv(
        f"data/game_data_{pd.Timestamp.now()}.csv",
        mode="a",
        header=not os.path.exists("data/game_log.csv"),
        index=False,
    )

    os.makedirs("models", exist_ok=True)

    for i, advisor in enumerate(ai_advisors):
        model_path = f"models/ai_advisor_{i}.pkl"
        advisor.save_model(model_path)

    print("Game over.")


if __name__ == "__main__":
    for _ in range(5):
        main()