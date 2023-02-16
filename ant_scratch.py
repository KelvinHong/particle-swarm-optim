# Algorithm from
# https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms
# and 
# http://www.scholarpedia.org/article/Ant_colony_optimization

import numpy as np
from scipy.spatial import distance_matrix
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import imageio
import os

class Map():
    def __init__(self, points):
        self.points = points
        self.n = points.shape[0]
        self.dm = distance_matrix(self.points, self.points)
        

class Ant():
    def __init__(self, num_points):
        self.npoints = num_points
        # Initialize route (TSP), zero based.
        self.route = np.random.permutation(self.npoints)
        self.current = self.route[-1]

class AntColony():
    def __init__(self, num_ant: int, colony_map: Map, rho = 0.5, q = 10, alpha=0.5, 
                beta=2, phe_max = 20):
        # rho is pheromone vaporate rate, q is a constant for calculating
        # pheromone.
        self.map = colony_map
        self.num_ant = num_ant
        self.ants = [Ant(self.map.n) for _ in range(self.num_ant)]
        # Generate random symmetric matrix for pheromones (tau)
        self.phe_max = phe_max # Set max pheromone level
        self.pheromones = np.random.uniform(size = (self.map.n, self.map.n))
        self.pheromones = 0.5 * (self.pheromones + self.pheromones.transpose())
        np.fill_diagonal(self.pheromones, 0)
        self.rho = rho
        self.q = q
        self.alpha = alpha
        self.beta = beta
        # Attractiveness Maps
        self.attract = 1 / self.map.dm
        np.fill_diagonal(self.attract, 0)

    @property
    def pheromones(self):
        return self._pheromones
    
    @pheromones.setter
    def pheromones(self, value):
        # Always clip pheromones when it is changed.
        self._pheromones = np.clip(value, 0, 20)

    def generateProb(self):
        # Calculate probabilities
        composite_mat = np.power(self.pheromones, self.alpha) * \
                    np.power(self.attract, self.beta)
        composite_vec = composite_mat.sum(axis=1)
        self.probs = composite_mat / np.expand_dims(composite_vec, axis=1)

    def generateSol(self):
        # This be called after probability is calculated.
        # Let every ant traverse the graph again, which means generate 
        # the permutation, but with probability instead of randomly.
        assert hasattr(self, "probs"), "Please run generateProb() before running this method."
        for ant in self.ants:
            current_route = [ant.current]
            # Keep adding next node until every node is visited.
            while len(current_route) < self.map.n:
                current_node = current_route[-1]
                mn_probs = deepcopy(self.probs[current_node, :])
                # Set visited nodes probability to zero.
                mn_probs[current_route] = 0
                # Normalize probability array as per numpy multinomial requirement
                mn_probs = mn_probs / np.sum(mn_probs)
                # Perform single time multinomial sampling.
                next_node = np.argmax(np.random.multinomial(1, mn_probs))
                current_route.append(next_node)
                ant.current = next_node
            ant.route = deepcopy(current_route)


    def updatePheromones(self):
        # Based on route from each ant, update pheromones.
        ## First evaporate existing pheromones based on 
        ## evaporate coefficient (rho)
        self.pheromones = (1-self.rho) * self.pheromones
        ## Update pheromones from each ant.
        for ant in self.ants:
            # Ant Pheromone Contribution initialized as zeros.
            pheromone_contrib = np.zeros((self.map.n, self.map.n))
            start_ps = ant.route[:]
            end_ps = np.concatenate((ant.route[1:], ant.route[:1]))
            ant_travel_total = np.sum(self.map.dm[start_ps, end_ps])
            pheromone_contrib[start_ps, end_ps] = \
                self.q / ant_travel_total
            ### The code below will lead to kind of greed algorithm
            # for curr_node, next_node in zip(ant.route, ant.route[1:]):
            # pheromone_contrib[
            #     [curr_node, next_node], 
            #     [next_node, curr_node]
            # ] = self.q / self.map.dm[curr_node, next_node]
            self.pheromones += pheromone_contrib

    def evaluate(self, **kwargs):
        # Calculate the distance travelled by each ant
        # Aggregate function can be applied, if none, return a list of distances,
        # A provided function should be able to take an array and return a scalar.
        # Like Sum, Mean, Min or Max.
        # Example usage: provide
        # self.evaluate(min=np.min, max=np.max, mean=np.mean)
        distances = []
        for ant in self.ants:
            distances.append(np.sum(self.map.dm[ant.route[:-1], ant.route[1:]]))
        optimal = np.argmin(distances)
        self.current_best_route = self.ants[optimal].route
        if kwargs:
            return {
                key: func(distances) for key, func in kwargs.items()
            }
        else:
            return distances
        
    def visualize(self, title = None, remain = False, save_as = None):
        cm = mplcm.get_cmap('hot_r')
        # Visualize nodes and paths based on pheromones density
        plt.clf()
        plt.xlim(-6, 4)
        plt.ylim(-4, 6)
        # Plot lines (paths)
        for start in range(self.map.n-1):
            for end in range(start+1, self.map.n):
                spoint = self.map.points[start]
                epoint = self.map.points[end]
                # Calculate pheromone level normalized to 0 - 1.
                ph_level = self.pheromones[start, end] / self.phe_max
                color = cm(ph_level)
                plt.plot([spoint[0], epoint[0]], [spoint[1], epoint[1]], color=color, 
                        linestyle = '--', linewidth=5 * ph_level, zorder=0)
                plt.plot([spoint[0], epoint[0]], [spoint[1], epoint[1]], color=(*(color[:3]), 0.4), 
                        linestyle = '--', linewidth=10 * ph_level, zorder=0)
        # Plot nodes
        plt.scatter(self.map.points[:, 0], self.map.points[:, 1], s=200, color="green", zorder=1)
        # Plot number on nodes
        for n in range(self.map.n):
            plt.text(*self.map.points[n], str(n), color="white",
                horizontalalignment='center', verticalalignment='center',)
        # Plot current best route
        cbr_start_ind = self.current_best_route[:]
        cbr_end_ind = np.concatenate((self.current_best_route[1:], self.current_best_route[:1]))
        cbr_start = self.map.points[cbr_start_ind]
        cbr_end = self.map.points[cbr_end_ind]
        cbr_dir = cbr_end - cbr_start
        plt.quiver(cbr_start[:, 0], cbr_start[:, 1], cbr_dir[:, 0], cbr_dir[:, 1], 
            angles='xy', scale_units='xy', scale=1.2, width=0.01,  
            color="green", alpha=0.6, zorder=2)
        if title is not None:
            plt.title(title)
        if save_as is not None:
            plt.savefig(save_as, 
                transparent = False,  
                facecolor = 'white'
            )
        if not remain:
            plt.pause(1e-1)
        else:
            plt.savefig("./currentColony.png")
            plt.waitforbuttonpress()

if __name__ == "__main__":
    os.makedirs("./gif/", exist_ok=True)
    np.random.seed(1)
    # Initialize maps with nodes
    # Distances are the L2 norm on the plane
    num_ant = 70
    phe_q = 1000 / num_ant
    epochs = 100
    # Random map
    points = np.array([
        [-5, 1],
        [-3, 5],
        [-5, -1],
        [-2, 2],
        [2, 0],
        [3, 1],
        [2, -2.5],
        [3, -3],
        [-1, -1.5],
        [0,4],
        [-5.5, 3],
    ])
    # Rectangle
    # points = np.array([
    #     [3, 3],
    #     [3, 1],
    #     [3, -1],
    #     [3, -3],
    #     [1, -3],
    #     [-1, -3],
    #     [-3, -3],
    #     [-5, -3],
    #     [-5, -1],
    #     [-5, 1],
    #     [-5, 3],
    #     [-3, 3],
    #     [-1, 3],
    #     [1, 3],
    # ])
    num_points = points.shape[0]
    colony_map = Map(points)
    ac = AntColony(num_ant, colony_map, q = phe_q)
    plt.ion()
    plt.show()
    for e in tqdm(range(epochs)):
        img_save_as = f"./gif/frame_{e}.png"
        ac.generateProb()
        ac.generateSol()
        ac.updatePheromones()
        print(ac.evaluate(min=np.min, mean=np.mean, max=np.max))
        print("Current best route: ", ac.current_best_route)
        if e == epochs-1:
            ac.visualize(title = f"Epoch {e+1} (Darker path indicates heavier pheromones,\npress anykey to close)", 
                        remain = True,
                        save_as=img_save_as)
        else:
            ac.visualize(title = f"Epoch {e+1} (Darker path indicates heavier pheromones)",
                        save_as=img_save_as)
    # Save animation as gif
    frames = []
    for e in range(epochs):
        image = imageio.v2.imread(f'./gif/frame_{e}.png')
        frames.append(image)
    imageio.mimsave('./AntColony.gif', # output gif
                frames,          # array of input frames
                fps = 5,
                loop=1)         # optional: frames per second