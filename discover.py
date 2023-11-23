import glob
import os
import random

import numpy as np
import os.path as op

from PIL import Image

from config import SAVE_PATH, NUM_OF_MOVES, POPULATION_SIZE, NUM_OF_CHROMOSOMES_IN_ONE_SET, DROP_COUNT, MUTATION_RATE, \
    MUTATION_COUNT, CH_COLORS, GRID_SIZE, STARTING_X, STARTING_Y, MAX_GENERATION
from environment.entities import Field, Generator, Moves, Chromosome, ChromosomeSet
from utils.geometry import Point

np.random.seed(123)


def create_gif():
    image_abs_paths = sorted(glob.glob(op.join(SAVE_PATH, "*.jpg")))
    images = []
    for image_abs_path in image_abs_paths:
        image = Image.open(image_abs_path)
        image = image.resize((512, 512))
        images.append(image)
    images[0].save('Animation.gif',
                   save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)


if __name__ == '__main__':
    os.makedirs(SAVE_PATH, exist_ok=True)
    field = Field(num_of_rows=GRID_SIZE, num_of_cols=GRID_SIZE)
    starting_points = [Point(x=STARTING_X, y=STARTING_Y)]
    generator = Generator(num_of_parameters=NUM_OF_MOVES,
                          values=Moves.get_all(),
                          starting_points=starting_points)
    chromosome_sets = generator.generate_chromosomes(size=POPULATION_SIZE,
                                                     set_size=NUM_OF_CHROMOSOMES_IN_ONE_SET)

    success_generation_no = 0
    num_of_generation = 1
    fitnesses = np.array(list(map(lambda ch: ch.calculate_fitness(field=field), chromosome_sets)))
    max_fitness = -np.inf
    while num_of_generation < MAX_GENERATION:
        drop_idxs = np.argsort(fitnesses)[:DROP_COUNT]
        best_chromosome_set = chromosome_sets[np.argmax(fitnesses)]
        if max(fitnesses) > max_fitness:
            max_fitness = max(fitnesses)
            print(f"New max: {max_fitness}")
            image = field.visualize(chromosome_set=best_chromosome_set)
            image.save(op.join(SAVE_PATH, f"{success_generation_no:04d}.jpg"))
            success_generation_no += 1
            num_of_generation = 0

        fitnesses = np.delete(fitnesses, drop_idxs)
        chromosome_sets = np.delete(chromosome_sets, drop_idxs)
        normalized_fitnesses = fitnesses / sum(fitnesses)
        new_chromosome_sets = []
        for _ in range(DROP_COUNT):
            operation = random.random()
            if operation < MUTATION_RATE:
                will_be_mutated = random.choices(chromosome_sets, k=1)[0]
                mutating_chromosome = np.random.randint(0, NUM_OF_CHROMOSOMES_IN_ONE_SET, MUTATION_COUNT)
                mutate_indexes = np.random.randint(0, NUM_OF_MOVES, MUTATION_COUNT)
                for ch_no, idx in zip(mutating_chromosome, mutate_indexes):
                    will_be_mutated.chromosomes[ch_no].directions[idx] = Moves.get_random()
                new_chromosome_sets.append(will_be_mutated)
            else:
                set_1, set_2 = random.choices(chromosome_sets, weights=normalized_fitnesses, k=2)
                chromosomes = []
                for ch_no, ch_color in zip(range(NUM_OF_CHROMOSOMES_IN_ONE_SET), CH_COLORS):
                    split_point = np.random.randint(0, NUM_OF_MOVES)
                    directions = set_1.chromosomes[ch_no].directions[:split_point]
                    directions.extend(set_2.chromosomes[ch_no].directions[split_point:])
                    chromosomes.append(Chromosome(directions=directions,
                                                  starting_point=set_1.chromosomes[ch_no].starting_point,
                                                  color=ch_color,
                                                  chromosome_no=ch_no))
                new_chromosome_sets.append(ChromosomeSet(chromosomes=chromosomes))
        chromosome_sets = np.append(chromosome_sets, new_chromosome_sets)
        fitnesses = np.array(list(map(lambda ch: ch.calculate_fitness(field=field), chromosome_sets)))
        num_of_generation += 1
        print("Generation: ", num_of_generation, " Max Fitness: ", max_fitness)
    # create_gif()
