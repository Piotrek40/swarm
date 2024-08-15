import torch
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import numpy as np

def calculate_metrics(outputs, targets, problem_type):
    """
    Calculates various metrics based on the problem type.

    Args:
        outputs (torch.Tensor): The model's outputs.
        targets (torch.Tensor): The true target values.
        problem_type (str): The type of problem ('classification' or 'regression').

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    if problem_type == 'classification':
        pred = np.argmax(outputs, axis=1)
        return {
            'accuracy': accuracy_score(targets, pred),
            'f1_score': f1_score(targets, pred, average='weighted')
        }
    elif problem_type == 'regression':
        return {
            'mse': mean_squared_error(targets, outputs),
            'rmse': np.sqrt(mean_squared_error(targets, outputs)),
            'r2_score': r2_score(targets, outputs)
        }
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

def calculate_model_complexity(model):
    """
    Calculates the complexity of a model based on its number of parameters.

    Args:
        model (torch.nn.Module): The model to analyze.

    Returns:
        int: The total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())

def calculate_diversity(population):
    """
    Calculates the diversity of a population of models.

    Args:
        population (list): A list of models.

    Returns:
        float: A measure of population diversity.
    """
    complexities = [calculate_model_complexity(model) for model in population]
    return np.std(complexities) / np.mean(complexities)

def calculate_ensemble_performance(ensemble, inputs, targets, problem_type):
    """
    Calculates the performance of an ensemble of models.

    Args:
        ensemble (list): A list of models forming the ensemble.
        inputs (torch.Tensor): The input data.
        targets (torch.Tensor): The true target values.
        problem_type (str): The type of problem ('classification' or 'regression').

    Returns:
        dict: A dictionary containing the ensemble's performance metrics.
    """
    outputs = torch.stack([model(inputs) for model in ensemble])
    ensemble_output = torch.mean(outputs, dim=0)
    return calculate_metrics(ensemble_output, targets, problem_type)

def track_fitness_over_time(swarm, generations):
    """
    Tracks the best fitness of a swarm over multiple generations.

    Args:
        swarm (EvolutionarySwarm): The swarm to track.
        generations (int): The number of generations to track.

    Returns:
        list: A list of best fitness values for each generation.
    """
    fitness_history = []
    for _ in range(generations):
        swarm.evolve()
        fitness_history.append(swarm.get_best_model().fitness)
    return fitness_history

def calculate_pareto_front(population, objectives):
    """
    Calculates the Pareto front for a population based on multiple objectives.

    Args:
        population (list): A list of models.
        objectives (list): A list of objective functions to minimize.

    Returns:
        list: The models on the Pareto front.
    """
    pareto_front = []
    for model in population:
        is_dominated = False
        model_objectives = [obj(model) for obj in objectives]
        for other_model in population:
            if model != other_model:
                other_objectives = [obj(other_model) for obj in objectives]
                if all(o1 <= o2 for o1, o2 in zip(other_objectives, model_objectives)) and any(o1 < o2 for o1, o2 in zip(other_objectives, model_objectives)):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_front.append(model)
    return pareto_front