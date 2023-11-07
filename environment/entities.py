import copy
import itertools
from enum import Enum
from typing import List
import numpy as np

from config import DROP_COUNT, SPLITTER_WIDTH, CELL_WIDTH, SPLITTER_COLOR, CELL_COLOR, NUM_OF_CHROMOSOMES_IN_ONE_SET, \
    POPULATION_SIZE, CH_COLORS
from utils.geometry import Point
import math

import numpy as np

from utils.geometry import BoundingBox
from utils.visualizer import Visualizer

class Field:
    def __init__(self, num_of_rows, num_of_cols):
        self.num_of_rows = num_of_rows
        self.num_of_cols = num_of_cols
        self.game = np.zeros((self.num_of_rows, self.num_of_cols))
        self.width_of_game = (num_of_rows + 1) * SPLITTER_WIDTH + self.num_of_rows * CELL_WIDTH
        self.height_of_game = (num_of_cols + 1) * SPLITTER_WIDTH + self.num_of_rows * CELL_WIDTH
        self.image = None

    def get_visit_map(self, chromosome_set: 'ChromosomeSet'):
        visit_map = {}
        for chromosome in chromosome_set.chromosomes:
            for point in chromosome.visited_cells:
                key = f"{point.x},{point.y}"
                if key not in visit_map.keys():
                    visit_map[key] = []
                visit_map[key].append(chromosome)
        return visit_map

    def visualize(self, chromosome_set: 'ChromosomeSet'):
        self.image = Visualizer.create(width=self.width_of_game, height=self.height_of_game, fill=SPLITTER_COLOR)
        for row in range(self.num_of_rows):
            for col in range(self.num_of_cols):
                cell_point = Point(x=row, y=col)
                cell_box = self.get_box(point=cell_point)
                self.image.draw_bbox(cell_box, color=CELL_COLOR, label="", fill=False)
        visit_map = self.get_visit_map(chromosome_set=chromosome_set)
        for visited_cell, visit_chromosomes in visit_map.items():
            x, y = visited_cell.split(',')
            point = Point(x=x, y=y)
            if not Chromosome.is_out_of_map(point, self):
                box = self.get_box(point)
                if len(visit_chromosomes) == 1:
                    self.image.draw_bbox(box, fill=True, color=visit_chromosomes[0].color, label="")
                elif len(visit_chromosomes) == 2:
                    box_left = copy.deepcopy(box)
                    box_right = copy.deepcopy(box)
                    box_left.xmax = box.center.x
                    box_right.xmin = box_right.center.x
                    self.image.draw_bbox(box_left, fill=True, color=visit_chromosomes[0].color, label="")
                    self.image.draw_bbox(box_right, fill=True, color=visit_chromosomes[1].color, label="")
                elif len(visit_chromosomes) == 3:
                    box_left = copy.deepcopy(box)
                    box_mid = copy.deepcopy(box)
                    box_right = copy.deepcopy(box)
                    box_left.xmax = box.xmin + (box.width // 3)
                    box_mid.xmin = box_left.xmax
                    box_mid.xmax = box_left.xmax + (box.width // 3)
                    box_right.xmin = box_mid.xmax
                    self.image.draw_bbox(box_left, fill=True, color=visit_chromosomes[0].color, label="")
                    self.image.draw_bbox(box_mid, fill=True, color=visit_chromosomes[1].color, label="")
                    self.image.draw_bbox(box_right, fill=True, color=visit_chromosomes[2].color, label="")
                elif len(visit_chromosomes) == 4:
                    box_left_top = copy.deepcopy(box)
                    box_left_bottom = copy.deepcopy(box)
                    box_right_top = copy.deepcopy(box)
                    box_right_bottom = copy.deepcopy(box)
                    box_left_top.xmax = box.xmin + (box.width // 2)
                    box_left_top.ymax = box.ymin + (box.height // 2)
                    box_left_bottom.xmax = box.xmin + (box.width // 2)
                    box_left_bottom.ymin = box.ymin + (box.height // 2)
                    box_right_top.xmin = box.xmin + (box.width // 2)
                    box_right_top.ymax = box.ymin + (box.height // 2)
                    box_right_bottom.xmin = box.xmin + (box.width // 2)
                    box_right_bottom.ymin = box.ymin + (box.height // 2)
                    self.image.draw_bbox(box_left_top, fill=True, color=visit_chromosomes[0].color, label="")
                    self.image.draw_bbox(box_left_bottom, fill=True, color=visit_chromosomes[1].color, label="")
                    self.image.draw_bbox(box_right_top, fill=True, color=visit_chromosomes[2].color, label="")
                    self.image.draw_bbox(box_right_bottom, fill=True, color=visit_chromosomes[3].color, label="")

            center = box.center.y
            box.ymin = center - 7
            box.ymax = center + 7
            self.image.draw_bbox(box, fill=True, color=CELL_COLOR, label="")
        for chromosome_no, chromosome in enumerate(chromosome_set.chromosomes):
            self.visualize_single(chromosome=chromosome, chromosome_no=chromosome_no)
            # self.image._put_text(label=f"Cost: {cost}", x=20, y=chromosome_no * 30, outline_color=chromosome.color,
            #                      rotation=0, font_size=20)
        self.image.show_image(delay=1)
        return self.image.image

    def visualize_single(self, chromosome: 'Chromosome', chromosome_no: int):
        prev_position = chromosome.starting_point
        prev_box = self.get_box(prev_position)
        prev_center = prev_box.center
        prev_center.x = prev_box.xmax - ((64 // NUM_OF_CHROMOSOMES_IN_ONE_SET) * (chromosome.chromosome_no % POPULATION_SIZE)) - 10
        prev_center.y -= 3
        self.image.draw_point(point=prev_center, color=chromosome.color, label="")
        for idx, direction in enumerate(chromosome.directions):
            current_position = Point(x=prev_position.x + direction.value.x,
                                     y=prev_position.y + direction.value.y)
            current_box = self.get_box(current_position)
            current_center = current_box.center
            prev_box = self.get_box(prev_position)
            prev_center = prev_box.center
            prev_center.x = prev_box.xmax - ((64 // NUM_OF_CHROMOSOMES_IN_ONE_SET) * (chromosome.chromosome_no % POPULATION_SIZE)) - 10
            prev_center.y -= 3
            current_center.x = current_box.xmax - ((64 // NUM_OF_CHROMOSOMES_IN_ONE_SET) * (chromosome.chromosome_no % POPULATION_SIZE)) - 10
            current_center.y -= 3
            mid = Point.mid_point_of(current_center, prev_center)
            self.image.draw_point(point=current_center, color=chromosome.color, label="")
            self.image.draw_line(from_point=prev_center,
                                 to_point=current_center, color=chromosome.color, label="")
            self.image._put_text(label=idx, x=mid.x, y=mid.y, rotation=0, outline_color=SPLITTER_COLOR,
                                 text_color=(255, 255, 255))
            prev_position = current_position

    def get_box(self, point: Point):
        xmin = (point.x * CELL_WIDTH) + (SPLITTER_WIDTH * (point.x + 1))
        ymin = (point.y * CELL_WIDTH) + (SPLITTER_WIDTH * (point.y + 1))
        xmax = ((point.x + 1) * CELL_WIDTH) + (SPLITTER_WIDTH * (point.x + 1))
        ymax = ((point.y + 1) * CELL_WIDTH) + (SPLITTER_WIDTH * (point.y + 1))
        cell_box = BoundingBox(xmin=xmin, ymin=ymin,
                               xmax=xmax, ymax=ymax)
        return cell_box


class Moves(Enum):
    L = Point(x=-1, y=0)
    LT = Point(x=-1, y=-1)
    T = Point(x=0, y=-1)
    RT = Point(x=1, y=-1)
    R = Point(x=1, y=0)
    RB = Point(x=1, y=1)
    B = Point(x=0, y=1)
    RL = Point(x=-1, y=1)

    @staticmethod
    def get_all():
        return [Moves.L,
                Moves.LT,
                Moves.T,
                Moves.RT,
                Moves.R,
                Moves.RB,
                Moves.B,
                Moves.RL]

    @staticmethod
    def get_random():
        moves = Moves.get_all()
        return moves[np.random.randint(0, len(moves))]


class ChromosomeSet:
    def __init__(self, chromosomes: List['Chromosome']):
        self.chromosomes = chromosomes

    def calculate_fitness(self, field):
        total_fitness = 0
        pre_visited_points = []
        for chromosome in self.chromosomes:
            chromosome.visited_cells = [chromosome.starting_point]
            current_position = chromosome.starting_point
            prev_direction = None
            for direction in chromosome.directions:
                if prev_direction is not None:
                    unit_vector_1 = [direction.value.x, direction.value.y] / np.linalg.norm(
                        [direction.value.x, direction.value.y])
                    unit_vector_2 = [prev_direction.value.x, prev_direction.value.y] / np.linalg.norm(
                        [prev_direction.value.x, prev_direction.value.y])
                    dot_product = np.dot(unit_vector_1, unit_vector_2)
                    angle = math.degrees(np.arccos(dot_product))
                    total_fitness -= abs((angle // 45) * 20)
                current_position = Point(x=current_position.x + direction.value.x,
                                         y=current_position.y + direction.value.y)
                chromosome.visited_cells.append(current_position)
                if current_position in pre_visited_points:
                    total_fitness -= 20
                else:
                    pre_visited_points.append(current_position)
                    total_fitness += 50
                if Chromosome.is_out_of_map(current_position, field):
                    total_fitness -= 80
                prev_direction = direction
        return total_fitness


class Chromosome:
    def __init__(self, directions, starting_point: Point, color, chromosome_no: int):
        self.directions = directions
        self.starting_point = starting_point
        self.color = color
        self.chromosome_no = chromosome_no
        self.visited_cells = None
        self.fitness = 0

    @staticmethod
    def is_out_of_map(pos, field):
        if not 0 <= pos.x < field.num_of_rows or not 0 <= pos.y < field.num_of_cols:
            return True


class Generator:

    def __init__(self, num_of_parameters: int, values: List, starting_points: List):
        self.num_of_parameters = num_of_parameters
        self.values = values
        self.starting_points = starting_points

    def generate_chromosomes(self, size, set_size) -> List[ChromosomeSet]:
        chromosome_sets = []
        for s in range(size):
            chromosome_set = []
            ch_no = 0
            for c, ch_color in zip(range(set_size), CH_COLORS):
                directions = []
                for p in range(self.num_of_parameters):
                    random_indx = np.random.randint(0, len(self.values))
                    directions.append(self.values[random_indx])
                random_indx = np.random.randint(0, len(self.starting_points))
                chromosome_set.append(Chromosome(directions=directions,
                                                 starting_point=self.starting_points[random_indx],
                                                 color=ch_color,
                                                 chromosome_no=ch_no))
                ch_no += 1
            chromosome_sets.append(ChromosomeSet(chromosomes=chromosome_set))
        return np.array(chromosome_sets)


def get_best_n(chromosomes, n, field):
    combination_list = list(itertools.combinations(chromosomes, n))
    fitnesses = []
    for comb in combination_list:
        visit_map = field.get_visit_map(chromosomes=comb)
        comb_fitness = sum(list(map(lambda ch: ch.fitness, comb)))
        for pt, chs in visit_map.items():
            x, y = pt.split(',')
            point = Point(x=x, y=y)
            if point != chs[0].starting_point and len(chs) > 1:
                comb_fitness -= ((len(chs) - 1) * 5)
            elif point != chs[0].starting_point and len(chs) == 1:
                comb_fitness += 20
        fitnesses.append(comb_fitness)
    best_idxs = np.argsort(fitnesses)[::-1]
    hold_chromosome_nos = set()
    for best_idx in best_idxs:
        chs = combination_list[best_idx]
        for ch in chs:
            hold_chromosome_nos.add(ch.chromosome_no)
        if len(chromosomes) - len(hold_chromosome_nos) <= DROP_COUNT:
            break
    return combination_list[best_idxs[0]], list(hold_chromosome_nos), max(fitnesses)
