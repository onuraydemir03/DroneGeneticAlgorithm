import seaborn as sns
import random

GRID_SIZE = 9
STARTING_X = 0
STARTING_Y = 8

NUM_OF_MOVES = 10
NUM_OF_CHROMOSOMES_IN_ONE_SET = 4

MAX_GENERATION = 1000

CELL_WIDTH = 64
SPLITTER_WIDTH = 12
CELL_COLOR = (255, 251, 245)
SPLITTER_COLOR = (129, 143, 180)
POS_COLOR = (0, 255, 0)
CH_COLORS = [
    (60, 42, 33),
    (242, 247, 161),
    (225, 152, 152),
    (241, 90, 89)
]

# Choose a palette from seaborn (e.g., 'Set3')
palette = sns.color_palette('Set3')

POPULATION_SIZE = 50
MUTATION_RATE = 0.01
MUTATION_COUNT = int(0.05 * NUM_OF_MOVES)
DROP_BAD_RATE = 0.2
DROP_COUNT = int(DROP_BAD_RATE * POPULATION_SIZE)
CROSSOVER_RATE = 1 - MUTATION_RATE
SAVE_PATH = f"./Generations2"
for cl in range(NUM_OF_CHROMOSOMES_IN_ONE_SET - len(CH_COLORS)):
    random_color = random.choice(palette)
    random_color = tuple([int(cl * 255) for cl in random_color])
    CH_COLORS.append(random_color)