import numpy as np
import cv2
from random import choice
import os
import uuid
from math import sqrt
from random import uniform
from multiprocessing import Process
from create_dataset_compatible_with_map_factory import MapSample
from tqdm import tqdm


class MapFactory(object):
    MAZE_COMBINATIONS = (
        (7, 0.06),
        (7, 0.07),
        (7, 0.08),
        (7, 0.09),
        (5, 0.10),
        (5, 0.11),
        (5, 0.12),
        (5, 0.13),
        (5, 0.14),
        # (3, 0.35),
        # (3, 0.36),
        # (3, 0.37),
        # (3, 0.38),
        # (3, 0.39),
        # (3, 0.40),
    )

    @staticmethod
    def random_maze(h=100, w=100, k=7, alpha=0.06):
        maze = np.random.rand(h, w)
        maze = np.where(maze > alpha, 1, 0)
        kernel = np.ones((k, k), np.uint8)
        maze = cv2.morphologyEx(maze.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
        return maze

    @classmethod
    def random_maze_(cls, h=100, w=100):
        k, alpha = choice(cls.MAZE_COMBINATIONS)
        return MapFactory.random_maze(h, w, k, alpha)
    
    @staticmethod
    def random_start_goal(maze, min_dist):
        h, w = maze.shape
        max_it = h * w
        start, goal = np.array([[0], [0]], dtype=np.int64), np.array([[0], [0]], dtype=np.int)
        i = 0
        while np.linalg.norm(start - goal) < min_dist and i < max_it:
            x = np.random.randint(0, h, size=(2, 1), dtype=np.int64)   # rows
            y = np.random.randint(0, w, size=(2, 1), dtype=np.int64)   # cols
            start = np.concatenate((x[0], y[0]))
            goal = np.concatenate((x[1], y[1]))
            if maze[start[0], start[1]] > 0 or maze[goal[0], goal[1]] > 0:
                start = goal
            i += 1
        if i == max_it:
            raise ValueError('Cannot find start and goal with enough distance')
        return start, goal

    @staticmethod
    def make_sample_(h, w, min_dist):
        map = MapFactory.random_maze_(h, w)
        start, goal = MapFactory.random_start_goal(map, min_dist)
        return map, start, goal

    @staticmethod
    def make_sample(h, w, min_dist, max_it, obst_margin, goal_margin):
        sample = None
        dmap, start, goal = MapFactory.make_sample_(h, w, min_dist)
        # solved, path = MapFactory.solve(map, start, goal, max_it, obst_margin, goal_margin)
        # if solved:
        sample = MapSample(dmap, start, goal)
        return sample


PATH = 'map_dataset'

def generate_sample(h, w, min_dist, max_ds_it, obst_margin, goal_margin, location=PATH):
    sample = MapFactory.make_sample(h, w, min_dist, max_ds_it, obst_margin, goal_margin)
    filename = os.path.join(location, str(uuid.uuid4()) + '.pt')
    sample.save(filename)

def generate_random_samples(h, w, min_dist_th, max_ds_it, obst_margin, goal_margin, n=10000, location=PATH):
    max_dist = sqrt(h ** 2 + w ** 2) - 1e-3
    for i in tqdm(range(n)):
        try:
            min_dist = uniform(min_dist_th, max_dist)
            generate_sample(h, w, min_dist, max_ds_it, obst_margin, goal_margin, location)
        except ValueError as e:
            print(e)
            continue


class RandomMapGenerator(object):
    def __init__(self, h, w, min_dist_th, max_ds_it, obst_margin, goal_margin):
        super(RandomMapGenerator, self).__init__()
        self.h = h
        self.w = w
        self.min_dist_th = min_dist_th
        self.max_ds_it = max_ds_it
        self.obst_margin = obst_margin
        self.goal_margin = goal_margin

    def execute(self, n=10000, n_process=4, location=PATH):
        if not os.path.exists(os.path.abspath(location)):
            os.mkdir(os.path.abspath(location))
        for _ in range(n_process):
            Process(target=generate_random_samples, args=(self.h, self.w, self.min_dist_th, self.max_ds_it, 
                self.obst_margin, self.goal_margin, n, location)).start()


if __name__ == '__main__':
    # sample = MapFactory.make_sample(50, 50, 25, 10000, 0, 1)
    # bgr = sample.bgr_map()
    rmg = RandomMapGenerator(100, 100, 20, 10000, 1, 0)
    rmg.execute(50, n_process=2)
