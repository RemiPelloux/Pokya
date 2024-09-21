import random

import torch

from ai.ai_advisor import AIAdvisor


def initialize_population(game, population_size):
    return [AIAdvisor(game, player_id=i, use_rl=True) for i in range(population_size)]


def evaluate_fitness(advisors, simulations=100):
    fitness_scores = []
    for advisor in advisors:
        wins = advisor.monte_carlo_simulation(
            advisor.game.get_hand(advisor.player_id), advisor.game.board, simulations
        )
        fitness_scores.append((advisor, wins))
    return fitness_scores


def select_parents(fitness_scores, num_parents):
    fitness_scores.sort(key=lambda x: x[1], reverse=True)
    return [advisor for advisor, score in fitness_scores[:num_parents]]


def crossover(parents, population_size):
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(parents, 2)
        child = AIAdvisor(parent1.game, player_id=len(new_population), use_rl=True)
        # Combine parts of parent models (e.g., neural network weights)
        for param1, param2, param_child in zip(
            parent1.model.parameters(),
            parent2.model.parameters(),
            child.model.parameters(),
        ):
            param_child.data.copy_(0.5 * param1.data + 0.5 * param2.data)
        new_population.append(child)
    return new_population


def mutate(advisors, mutation_rate=0.01):
    for advisor in advisors:
        for param in advisor.model.parameters():
            if random.random() < mutation_rate:
                param.data += torch.randn_like(param) * 0.1


def replace_population(old_population, new_population):
    return new_population


def genetic_algorithm(
    game, population_size=10, generations=50, num_parents=5, mutation_rate=0.01
):
    population = initialize_population(game, population_size)
    for generation in range(generations):
        fitness_scores = evaluate_fitness(population)
        parents = select_parents(fitness_scores, num_parents)
        new_population = crossover(parents, population_size)
        mutate(new_population, mutation_rate)
        population = replace_population(population, new_population)
    return population
