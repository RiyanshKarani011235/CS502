import matplotlib
import random
from tqdm import tqdm
import collections
import math

import datetime
from datetime import date
import pandas as pd
import numpy as np
from plotly import __version__
#%matplotlib inline

# INITIALIZE PLOTLY TO RUN OFFLINE INSTEAD OF ONLINE
import json
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from plotly import tools

import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
cf.go_offline()

init_notebook_mode(connected=True)

class AntClusteringModel:
    def __init__(self, grid_size=200, num_ants=10, num_objects=100, ant_memory_size=50, k1=0.1, k2=0.3, s=10, alpha=1, time_steps=5000000, num_key_frames=100):
        self._grid_size = grid_size
        self._num_ants = num_ants
        self._num_objects = num_objects
        self.k1 = k1
        self.k2 = k2
        self._s = s
        self._alpha = alpha
        self._time_steps = time_steps
        self._num_key_frames = num_key_frames
        self._grid = Grid(grid_size, ant_memory_size, k1, k2, s, alpha)

    def initialize_grid(self):
        self._grid.initialize_objects_randomly(self._num_objects)
        self._grid.initialize_ants_randomly(self._num_ants)

    def simulate(self):
        # make figure

        x,y = self._grid.get_objects()
        figure = {
            'data': [
                {'x': x, 'y': y, 'xaxis': 'x', 'yaxis': 'y', 'mode': 'markers', 'name': 'objects'},
                {'x': [0], 'y': [0], 'xaxis': 'x2', 'yaxis': 'y2', 'name': 'spatial entropy'}
            ],
            'layout': {
                'xaxis': {'anchor': 'y', 'domain': [0.0, 0.45]},
                'xaxis2': {'anchor': 'y2', 'domain': [0.55, 1.0]},
                'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0]},
                'yaxis2': {'anchor': 'x2', 'domain': [0.0, 1.0]}
            },
            'frames': []
        }

        # fill in most of layout
        figure['layout']['hovermode'] = 'closest'
        figure['layout']['updatemenus'] = [
            {
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 500, 'redraw': False},
                                 'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                        'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }
        ]

        spatial_entropy_x = []
        spatial_entropy_y = []
        for i in tqdm(range(self._time_steps)):
            self._grid.update()

            if i % 5000 == 0:
                spatial_entropy_x.append(i/50000)
                spatial_entropy_y.append(self._grid.compute_spatial_entropy(10))
                x,y = self._grid.get_objects()
                figure['frames'].append(
                    {'data': [
                        {'x': x, 'xaxis': 'x', 'y': y, 'yaxis': 'y', 'mode': 'markers', 'name': 'objects'},
                        {'x': spatial_entropy_x, 'y': spatial_entropy_y, 'xaxis': 'x2', 'yaxis': 'y2', 'name': 'spatial entropy'}
                    ]})

        self._figure = figure

    def plot(self):
        pyo.iplot(self._figure)

class Grid:
    def __init__(self, size, ant_memory_size, k1, k2, s, alpha):
        self._size = size
        self._ant_memory_size = ant_memory_size
        self._k1 = k1
        self._k2 = k2
        self._s = s
        self._alpha = alpha
        self.initialize_empty_grid()

    def initialize_empty_grid(self):
        self._grid = [[Cell() for i in range(self._size)] for j in range(self._size)]
    
    def initialize_objects_randomly(self, num_initialized_objects):
        initialized_objects = 0
        while initialized_objects < num_initialized_objects and initialized_objects < self._size*self._size:
            i = random.randint(0, self._size-1)
            j = random.randint(0, self._size-1)
            if not self._grid[i][j].has_object():
                # initialize objects with random labels
                self._grid[i][j].set_object(Object(random.randint(0, 1)))
                initialized_objects += 1
    
    def initialize_ants_randomly(self, num_initialized_ants):
        self._ants = []
        position_ants_map = {}
        initialized_ants = 0
        while initialized_ants < num_initialized_ants and initialized_ants < self._size*self._size:
            i = random.randint(0, self._size-1)
            j = random.randint(0, self._size-1)
            if (i, j) not in position_ants_map:
                self._ants.append(Ant(self._ant_memory_size, i, j, self._k1, self._k2))
                position_ants_map[(i, j)] = True
                initialized_ants += 1
                
    def get_neighbour_objects(self, i, j):
        i_min = i-self._s
        i_max = i+self._s
        j_min = j-self._s
        j_max = j+self._s

        if i_min < 0:
            i_min = 0
        if i_max >= self._size:
            i_max = self._size
        if j_min < 0:
            j_min = 0
        if j_max >= self._size:
            j_max = self._size

        neighbour_objects = []
        for k in range(i_min, i_max):
            for l in range(j_min, j_max):
                if self._grid[k][l].has_object():
                    neighbour_objects.append(self._grid[k][l].get_object())
                    
        return neighbour_objects
    
    def compute_frequency(self, i, j):
        f = 0
        obj = self._grid[i][j].get_object()
        for neigh_obj in self.get_neighbour_objects(i, j):
            f += (1 - obj.compute_dissimilarity(neigh_obj)) * 1.0 /self._alpha
        f /= (self._s ** 2)
        return f

    def update(self):
        for ant in self._ants:
            ant.move_randomly(self._size)

            i, j = ant.get_position()
            if ant.has_object():
                if not self._grid[i][j].has_object():
                    # ant already has object, decide whether to drop it
                    self._grid[i][j].set_object(ant.get_object())
                    f = self.compute_frequency(i, j)
                    p = 2 * f
                    if f >= self._k2:
                        p = 1
                    
                    if random.random() < p:
                        # drop
                        self._grid[i][j].set_object(ant.get_object())
                        ant.drop_object()
                    else:
                        self._grid[i][j].remove_object()

            elif self._grid[i][j].has_object():
                # decide wiether to pick up the object
                obj = self._grid[i][j].get_object()
                f = self.compute_frequency(i, j)
                p = (self._k1 / (self._k1 + f)) ** 2
                
                if random.random() < p:
                    # pick
                    obj = self._grid[i][j].get_object()
                    self._grid[i][j].remove_object()
                    ant.pick_object(obj)
                    
    def compute_spatial_entropy(self, s):
        spatial_entropy = 0
        num_objects = self.compute_num_objects()
        for i in range(0, self._size, s):
            for j in range(0, self._size, s):
                p = 0
                for k in range(s):
                    for l in range(s):
                        if self._grid[i+k][j+l].has_object():
                            p += 1
                p /= 1.0 * num_objects
                if p != 0:
                    spatial_entropy -= p * math.log(p)
        return spatial_entropy
                
    def compute_num_objects(self):
        num_objects = 0
        for i in range(self._size):
            for j in range(self._size):
                if self._grid[i][j].has_object():
                    num_objects += 1
        return num_objects
                    
    def get_objects(self):
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
                
        return x, y
    
    def __str__(self):
        return_value = []
        for i in range(self._size):
            return_value.append([])
            for j in range(self._size):
                if self._grid[i][j].has_object():
                    return_value[-1].append(self._grid[i][j].get_object())
                else:
                    return_value[-1].append(-1)
        return str(return_value)


class Cell:
    def __init__(self, object_=None):
        self._object = object_

    def has_object(self):
        return self._object != None

    def set_object(self, object_):
        self._object = object_

    def remove_object(self):
        self._object = None

    def get_object(self):
        return self._object
    
    def __str__(self):
        return str(self._t)

class Ant:
    def __init__(self, i, j, k1, k2, alpha):
        self.i = i
        self.j = j
        self._k1 = k1
        self._k2 = k2
        self._alpha = alpha
        self._object = None

    def has_object(self):
        return self._object != None
    
    def get_object(self):
        return self._object
    
    def pick_object(self, object_):
        self._object = object_
        
    def drop_object(self):
        self._object = None

    def get_position(self):
        return (self.i, self.j)

    def move_randomly(self, grid_size):
        delta_i = random.randint(-1, 1)
        delta_j = random.randint(-1, 1)
        if self.i + delta_i < grid_size and self.i + delta_i >= 0:
            self.i += delta_i

        if self.j + delta_j < grid_size and self.j + delta_j >= 0:
            self.j += delta_j

class Object:
    def __init__(self, t):
        self._t = t
    
    def compute_dissimilarity(self, obj):
        if self._t == obj._t:
            return 0
        return 1

model = AntClusteringModel(time_steps=50000)
model.initialize_grid()
print(model._grid)
model.simulate()
