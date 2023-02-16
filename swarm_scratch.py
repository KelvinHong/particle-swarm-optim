import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import imageio
import os
import numpy  as np

def clip(bound):
    return lambda x: x if abs(x) <= bound else bound * ((x > 0)*2-1)

class Particle():
    def __init__(self, x_bound, y_bound, vx_bound, vy_bound):
        # Initialize 2D particle.
        # All arguments should be positive
        self.x = random.random() * 2 * x_bound - x_bound
        self.y = random.random() * 2 * y_bound - y_bound
        self.vx = random.random() * 2 * vx_bound - vx_bound
        self.vy = random.random() * 2 * vy_bound - vy_bound
        self.pbest = self

    def __repr__(self):
        return f"x={self.x}, y={self.y}, vx={self.vx}, vy={self.vy}"

class Swarm():
    def __init__(self, n, objective, 
                x_bound, y_bound, vx_bound, vy_bound,
                w, phi_p, phi_g):
        self.x_bound, self.y_bound, self.vx_bound, self.vy_bound = \
            x_bound, y_bound, vx_bound, vy_bound
        self.swarm = [Particle(x_bound, y_bound, vx_bound, vy_bound) for _ in range(n)]
        self.objective = objective
        self.n = n
        # Minimizing the objective
        self.gbest_list = []
        self.calGroupBest()
        self.w = w
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.clip_x = clip(x_bound)
        self.clip_y = clip(y_bound)
        self.clip_vx = clip(vx_bound)
        self.clip_vy = clip(vy_bound)
        # Mesh grid for plotting contour plot 
        gbound_x = self.x_bound + self.vx_bound
        gbound_y = self.y_bound + self.vy_bound
        grid_finess = 51
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-gbound_x, gbound_x, grid_finess), 
                np.linspace(-gbound_y, gbound_y, grid_finess))
        self.contour_Z = objective(self.grid_X, self.grid_Y)
        self.ct_level = [0, 0.5, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

    def calGroupBest(self):
        self.scores = [self.objective(p.x, p.y) for p in self.swarm]
        if hasattr(self, "bestValue"):
            self.bestValue = min(self.bestValue, min(self.scores))
        else:
            self.bestValue = min(self.scores)
        if self.bestValue < min(self.scores):
            return
        gbest_index = self.scores.index(min(self.scores))
        self.gbest = deepcopy(self.swarm[gbest_index])
        self.gbest_list.append(deepcopy(self.swarm[gbest_index]))

    def step(self):
        for i in range(self.n):
            r_px, r_gx = random.random(), random.random()
            r_py, r_gy = random.random(), random.random()
            # Update Velocity
            self.swarm[i].vx = self.swarm[i].vx + \
                    self.phi_p * r_px * (self.swarm[i].pbest.x - self.swarm[i].x) + \
                    self.phi_g * r_gx * (self.gbest.x - self.swarm[i].x)
            self.swarm[i].vy = self.swarm[i].vy + \
                    self.phi_p * r_py * (self.swarm[i].pbest.y - self.swarm[i].y) + \
                    self.phi_g * r_gy * (self.gbest.y - self.swarm[i].y)
            # Clip velocity
            self.swarm[i].vx = self.clip_vx(self.swarm[i].vx)
            self.swarm[i].vy = self.clip_vy(self.swarm[i].vy)
            # Update Location
            self.swarm[i].x += self.swarm[i].vx
            self.swarm[i].y += self.swarm[i].vy
            # Clip Position
            self.swarm[i].x = self.clip_x(self.swarm[i].x)
            self.swarm[i].y = self.clip_y(self.swarm[i].y)
            # Update particle best
            current_value = self.objective(self.swarm[i].x, self.swarm[i].y)
            if current_value < \
                self.objective(self.swarm[i].pbest.x, self.swarm[i].pbest.y):
                self.swarm[i].pbest = self.swarm[i]
                # Update global best
                if current_value < self.bestValue:
                    self.gbest = self.swarm[i]
        self.calGroupBest()

    def visualize(self, title=None, remain=False, save_as = None):
        plt.clf()
        plt.contour(self.grid_X, self.grid_Y, self.contour_Z, self.ct_level, zorder=-2)
        plt.xlim(-self.x_bound-self.vx_bound, self.x_bound + self.vx_bound)
        plt.ylim(-self.y_bound-self.vy_bound, self.y_bound + self.vy_bound)
        for i in range(self.n):
            color = "red"
            plt.quiver(self.swarm[i].x, self.swarm[i].y, self.swarm[i].vx, self.swarm[i].vy, 
                color=color,  angles='xy', scale_units='xy', scale=2, width=0.005, zorder=-1)
        plt.plot([p.x for p in self.gbest_list], [p.y for p in self.gbest_list], 
                    linestyle="-", color="cyan", zorder = 0)
        plt.plot(self.gbest.x, self.gbest.y, "go", ms=6, zorder=1)
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
            plt.savefig("./currentSwarm.png")
            plt.waitforbuttonpress()

    def __str__(self):
        return "Swarm information: \n" + "".join([str(p) + "\n" for p in self.swarm])

# Design custom objective function that is to be minimized
# It is better to design so that x and y can be generic numpy arrays.
def himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

def rosenbrock(x, y):
    return (1-x) ** 2 + 100 * (y - x**2)**2

def beale(x, y):
    return (1.5-x+x*y)**2 + (2.25-x+x*y**2)**2 \
            + (2.625-x+x*y**3)**2

if __name__ == "__main__":
    os.makedirs("./gif/", exist_ok=True)
    np.random.seed(11120)
    epochs = 100
    objective = himmelblau
    # objective = rosenbrock
    # objective = beale
    s1 = Swarm(10, objective, 5, 5, 2, 2, 0.6, 2, 2)
    plt.ion()
    plt.show()
    for e in tqdm(range(epochs)):
        s1.step()
        img_save_as = f"./gif/frame_{e}.png"
        if e == epochs - 1:
            s1.visualize(title=f"PSO: Epoch {e+1} (Press anykey to close)", remain=True, 
                save_as = img_save_as)
        else:
            s1.visualize(title=f"PSO: Epoch {e+1}", save_as = img_save_as)
    # Save animation as gif
    frames = []
    for e in range(epochs):
        image = imageio.v2.imread(f'./gif/frame_{e}.png')
        frames.append(image)
    imageio.mimsave('./Swarm.gif', # output gif
                frames,          # array of input frames
                fps = 5,
                loop=1)         # optional: frames per second
    print(s1.gbest)
    print(s1.bestValue)