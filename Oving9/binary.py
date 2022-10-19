import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Gen:
    best_fitness: int
    avg_fitness: float

    def __init__(self, best, avg):
        self.best_fitness = best
        self.avg_fitness = avg


class Evolution():
    def __init__(self):
        self.generations = []
        self.start_gen = np.random.randint(0, 256, size=10)
        self.generations.append(self.start_gen)
        self.target = np.random.randint(0, 256)

    def mutation(self, x: int):
        rnd_int = np.random.randint(0, 256)
        return x ^ rnd_int

    def combine(self, x, y):
        floor = np.random.randint(0, 1)

        return math.floor((x + y) / 2) if floor == 1 else math.ceil((x + y) / 2)

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

        for i in np.random.randint(0, 9, size=3):
            new_gen[i] = self.mutation(new_gen[i])

        return new_gen

    def train(self):
        i = 0
        fs = []
        while True:

            fitnesses = [self.fitness(x) for x in self.generations[i]]
            fs.append(fitnesses)
            # Pick 5 best numbers
            gen = sorted(zip(fitnesses, self.generations[i]), reverse=True)[:5]
            if max(fitnesses) == 0:
                print(self.generations)
                return [Gen(max(f), sum(f)/len(f)) for f in fs], self.generations

            new_gen = self.get_new_gen(gen)
            print(new_gen)
            self.generations.append(new_gen)

            i += 1


def print_gens(gen, best, avg, i):
    print(f"Gen {i}: ")
    print(gen)
    print(f"Best Fitness: {best}")
    print(f"Avg Fitness: {avg}")
    print("")

evo = Evolution()

gen, gens = evo.train()

for i in range(len(gens)):
    print_gens(gens[i], gen[i].best_fitness, gen[i].avg_fitness, i)

print(f"Target: {evo.target}")