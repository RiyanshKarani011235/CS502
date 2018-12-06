import random
import collections
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from time import gmtime, strftime
import os

class ModelBasic:
    def __init__(self, grid_size=1000, num_ants=1000, num_objects=1000, ant_memory_size=1000, k1=0.4, k2=0.3):
        self._grid_size = grid_size
        self._num_ants = num_ants
        self._num_objects = num_objects
        self.k1 = k1
        self.k2 = k2
        self._grid = Grid(grid_size, ant_memory_size, k1, k2)

    def initialize_grid(self):
        self._grid.initialize_objects_randomly(self._num_objects)
        self._grid.initialize_ants_randomly(self._num_ants)

    def simulate(self):
        self._grid.show_grid()
        for i in tqdm(range(50000)):
            if i % 5000 == 0:
                self._grid.show_grid()

    def compute_similarity_measure(self, object, ant):
        pass

class Grid:
    def __init__(self, size, ant_memory_size, k1, k2):
        self._size = size
        self._ant_memory_size = ant_memory_size
        self._k1 = k1
        self._k2 = k2
        self.initialize_empty_grid()

    def initialize_empty_grid(self):
        self._grid = [[Cell() for i in range(self._size)] for j in range(self._size)]
    
    def initialize_objects_randomly(self, num_initialized_objects):
        initialized_objects = 0
        while initialized_objects < num_initialized_objects and initialized_objects < self._size*self._size:
            i = random.randint(0, self._size-1)
            j = random.randint(0, self._size-1)
            if not self._grid[i][j].has_object():
                self._grid[i][j].set_object(Object())
                initialized_objects += 1
    
    def initialize_ants_randomly(self, num_initialized_ants):
        self._ants = []
        position_ants_map = {}
        initialized_ants = 0
        while initialized_ants < num_initialized_ants and initialized_ants < self._size*self._size:
            i = random.randint(0, self._size-1)
            j = random.randint(0, self._size-1)
            if (i, j) not in position_ants_map:
                self._ants.append(Ant(self._ant_memory_size, i, j))
                position_ants_map[(i, j)] = True
                initialized_ants += 1

    def update(self):
        for ant in self._ants:
            ant.move_randomly(self._size)

            i, j = ant.get_position()
            # save in memory
            if self._grid[i][j].has_object():
                ant.object_encountered(1)
            else:
                ant.object_encountered(0)

            if ant.has_object():
                if not self._grid[i][j].has_object():
                    # ant already has object, decide whether to drop it
                    f = ant.compute_frequency()
                    p = (f/(f+self._k2))**2
                    if random.random() < p:
                        # drop
                        self._grid[i][j].set_object(Object())
                        ant.drop_object()
            elif self._grid[i][j].has_object():
                # decide wiether to pick up the object
                f = ant.compute_frequency()
                p = (self._k1/(f+self._k1))**2
                if random.random() < p:
                    # pick
                    self._grid[i][j].remove_object()
                    ant.pick_object()

    def show_grid(self):
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xticks(np.arange(0, 0, 1))
        ax.set_yticks(np.arange(0, 0, 1))

        x = []
        y = []
        for i in range(self._size):
            for j in range(self._size):
                if self._grid[i][j].has_object():
                    x.append(i)
                    y.append(j)

        for ant in self._ants:
            if ant.has_object():
                i, j = ant.get_position()
                x.append(i)
                y.append(j)

        plt.scatter(x, y, color='b', label='corpses')

        plt.legend()
        plt.grid(True)
        filepath = os.path.join(os.path.dirname(__file__), 'images')
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        filename = os.path.join(filepath, 'image_' + strftime("%y-%m-%d %h:%m:%s", gmtime()) + '.png')
        fig.savefig(filename)
        plt.close(fig)

class Cell:
    def __init__(self, object_=None, ant=None):
        self._object = object_
        self._ant = ant

    def has_object(self):
        return self._object != None

    def set_object(self, object_):
        self._object = object_

    def remove_object(self):
        self._object = None

    def get_object(self):
        return self._object

class Ant:
    def __init__(self, memory_size, i, j):
        self.memory_size = memory_size
        self.i = i
        self.j = j
        self._has_object = False
        self.memory = collections.deque(maxlen=self.memory_size)

    def has_object(self):
        return self._has_object

    def pick_object(self):
        self._has_object = True

    def drop_object(self):
        self._has_object = False

    def get_position(self):
        return (self.i, self.j)

    def move_randomly(self, grid_size):
        delta_i = random.randint(-1, 1)
        delta_j = random.randint(-1, 1)
        if self.i + delta_i < grid_size and self.i + delta_i >= 0:
            self.i += delta_i

        if self.j + delta_j < grid_size and self.j + delta_j >= 0:
            self.j += delta_j

    def object_encountered(self, object):
        self.memory.append(object)

    def compute_frequency(self):
        return self.memory.count(1)*1.0/self.memory_size

class Object:
    def __init__(self):
        pass

import sys
print(sys.version_info)

if __name__ == '__main__':
    model = ModelBasic()
    model.initialize_grid()
    model.simulate()
    print(model)

