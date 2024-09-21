import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the game log data from the CSV file
def load_game_log(file_path):
    return pd.read_csv(file_path)


# Plot player chip counts over time
def plot_player_chips(ax, data):
    for player_id in data["player_id"].unique():
        player_data = data[data["player_id"] == player_id]
        ax.plot(
            player_data["hand_number"],
            player_data["player_chips"],
            label=f"Player {player_id}",
        )
    ax.set_title("Player Chips Over Time", fontsize=14)
    ax.set_xlabel("Hand Number", fontsize=12)
    ax.set_ylabel("Player Chips", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(True)


# Plot the pot size over time
def plot_pot_size(ax, data):
    ax.plot(
        data["hand_number"].unique(),
        data.groupby("hand_number")["pot_size"].mean(),
        color="blue",
    )
    ax.set_title("Average Pot Size Over Time", fontsize=14)
    ax.set_xlabel("Hand Number", fontsize=12)
    ax.set_ylabel("Pot Size", fontsize=12)
    ax.grid(True)


# **New**: Create a pie chart for player states distribution
def plot_player_states_pie(ax, data):
    state_counts = data["player_state"].value_counts()
    ax.pie(
        state_counts,
        labels=state_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("pastel")[0: len(state_counts)],
    )
    ax.set_title("Distribution of Player States", fontsize=14)


# **New**: Create a stacked bar chart for player states per hand
def plot_player_states_stacked(ax, data):
    state_data = (
        data.groupby(["hand_number", "player_state"]).size().unstack(fill_value=0)
    )
    state_data.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", width=0.8)
    ax.set_title("Player States Per Hand (Stacked)", fontsize=14)
    ax.set_xlabel("Hand Number", fontsize=12)
    ax.set_ylabel("Count of Player States", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(True)


# Main function to visualize the game log
def visualize_game_log(file_path):
    data = load_game_log(file_path)

    # First Figure: Player Chips and Pot Size
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # 2 rows, 1 column
    plot_player_chips(ax1, data)
    plot_pot_size(ax2, data)
    plt.tight_layout()
    plt.show()


# show every csv file in the data folder
def show_all_csv_files():
    # show number associated with the file you want to visualize and pass it to visualize_game_log, we are already inside the data folder
    import os
    for i, file in enumerate(os.listdir()):
        if file.endswith(".csv"):
            print(f"{i}: {file}")
    file_number = int(input("Enter the number associated with the file you want to visualize: "))
    visualize_game_log(os.listdir()[file_number])


if __name__ == "__main__":
    show_all_csv_files()
