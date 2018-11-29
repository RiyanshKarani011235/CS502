class ModelBasic:
    def __init__(self, grid_size=1000, num_ants=10, num_objects=100):
        self.initialize_grid()
        self.initialize_ants()
        self.initialize_objects()

    '''
    initialize the grid
    '''
    def initialize_grid(self):
        pass

    '''
    initialize the ants
    '''
    def initialize_ants(self):
        pass

    '''
    initialize the objects
    '''
    def initialize_objects(self):
        pass

    def simulate(self):
        pass

    def compute_similarity_measure(self, object, ant):
        pass

class Ant:
    def __init__(self):
        pass

class Object:
    def __init__(self):
        pass