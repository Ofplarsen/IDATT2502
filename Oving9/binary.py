import math
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Gen:
    best_fitness: int
    avg_fitness: float

    def __init__(self, best, avg):
        self.best_fitness = best
        self.avg_fitness = avg


class Evolution():
    def __init__(self, max = 256, size=10):
        self.size = size
        self.max = max
        self.generations = []
        self.start_gen = np.random.randint(0, max, size=size)
        self.generations.append(self.start_gen)
        self.target = np.random.randint(0, max)

    def mutation(self, x: int):
        rnd_int = np.random.randint(0, self.max)
        return x ^ rnd_int

    def combine(self, x, y):
        floor = np.random.randint(0, 2)

        return math.floor((x + y) / 2) if floor == 1 else math.ceil((x + y) / 3)

    def fitness(self, x: int):
        return -np.abs(x - self.target)

    def get_new_combination(self, gen: zip):
        new_gen = []
        t = 0
        for i in gen:
            for y in gen:
                if t == 2:
                    t = 0
                    break
                new_gen.append(self.combine(i[1], y[1]))
                t += 1
        return new_gen

    def get_new_gen(self, old_gen):
        new_gen = self.get_new_combination(old_gen)

        for i in np.random.randint(0, self.size, size=math.floor(self.size/3)):
            new_gen[i] = self.mutation(new_gen[i])

        return new_gen

    def train(self):
        i = 0
        fs = []
        while True:

            fitnesses = [self.fitness(x) for x in self.generations[i]]
            fs.append(fitnesses)
            # Pick 5 best numbers
            gen = sorted(zip(fitnesses, self.generations[i]), reverse=True)[:(math.floor(self.size/2))]
            if max(fitnesses) == 0:
                print(self.generations)
                return [Gen(max(f), sum(f)/len(f)) for f in fs], self.generations

            new_gen = self.get_new_gen(gen)
            print(new_gen)
            self.generations.append(new_gen)

            i += 1

    def train_time(self):
        i = 0
        sts = []

        for t in range(3):
            print(i)
            st = time.time()
            while True:

                fitnesses = [self.fitness(x) for x in self.generations[i]]
                # Pick 5 best numbers
                gen = sorted(zip(fitnesses, self.generations[i]), reverse=True)[:(math.floor(self.size/2))]
                if max(fitnesses) == 0:
                    et = time.time()
                    sts.append(et-st)
                    break
                print(gen)
                new_gen = self.get_new_gen(gen)
                self.generations.append(new_gen)

                i += 1

        return sum(sts)/len(sts)


def print_gens(gen, best, avg, i):
    print(f"Gen {i}: ")
    print(gen)
    print(f"Best Fitness: {best}")
    print(f"Avg Fitness: {avg}")
    print("")

def task_1_1():
    evo = Evolution()

    gen, gens = evo.train()

    for i in range(len(gens)):
        print_gens(gens[i], gen[i].best_fitness, gen[i].avg_fitness, i)

    print(f"Target: {evo.target}")

def task_1_2():
    n = np.arange(8, 19)
    time  = []

    for i in range(8, 19):
        evo = Evolution(2**i, 10)
        time.append(evo.train_time())
        print(f"Done with: {2**i}")
    print(len(time))
    print(len(n))

    plt.plot(n, time)
    plt.show()



if __name__ == "__main__":
    #task_1_1()
    task_1_2()