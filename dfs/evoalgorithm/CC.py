# Cooperative Coevolution
import numpy as np
import math

# ハイパーパラメータ
WPOP_SIZE = 100         # 全体解集団のサイズ
PPOP_SIZE = 200         # 部分解集団のサイズ
WCROSSOVER_PROB = 0.9   # 全体解集団の交叉率
PCROSSOVER_PROB = 0.5   # 部分解集団の交叉率
WMUTATE_PROB = 0.05     # 全体解遺伝子の突然変異確率
PMUTATE_PROB = 0.1      # 部分解遺伝子の突然変異確率
WCHROM_LEN = 24         # 全体解個体のサイズ
PCHROM_LEN = 8          # 部分解集団のサイズ
TOURNAMENT_SIZE = 5     # トーナメントサイズ

# 部分解個体
class PartialIndividual:
    def __init__(self, wchrom_idx):
        if (wchrom_idx + 1) * PCHROM_LEN <= 32:
            self.chrom = [1] * PCHROM_LEN
        elif (wchrom_idx + 1) * PCHROM_LEN == 192:
            self.chrom = np.random.choice([0, 1], size=PCHROM_LEN, p=[0.8, 0.2])
            self.chrom[PCHROM_LEN-1] = 1
        else:
            # self.chrom = np.random.randint(0, 2, PCHROM_LEN)
            self.chrom = np.random.choice([0, 1], size=PCHROM_LEN, p=[0.8, 0.2])
        self.global_fitness = float('inf')

    def crossover(self, parent1, parent2, index1, index2):
        if index1 > index2:
            tmp = index1
            index1 = index2
            index2 = tmp
        for i in range(0, index1):
            self.chrom[i] = parent1.chrom[i]
        for i in range(index1, index2):
            self.chrom[i] = parent2.chrom[i]
        for i in range(index2, PCHROM_LEN):
            self.chrom[i] = parent1.chrom[i]
        self.mutate()
    
    def mutate(self):
        for i in range(PCHROM_LEN):
            if np.random.rand() < PMUTATE_PROB:
                self.chrom[i] = 1 - self.chrom[i]

# 部分解集団
class PartialPopulation:
    def __init__(self, wchrom_idx):
        self.population = []
        for i in range(PPOP_SIZE):
            individual = PartialIndividual(wchrom_idx)
            self.population.append(individual)
    
    def crossover(self):
        for i in range(int(PPOP_SIZE * (1 - PCROSSOVER_PROB)), PPOP_SIZE):
            # 二点交叉
            parent1 = min(np.random.choice(range(PPOP_SIZE), TOURNAMENT_SIZE), key=lambda x: self.population[x].global_fitness)
            parent2 = min(np.random.choice(range(PPOP_SIZE), TOURNAMENT_SIZE), key=lambda x: self.population[x].global_fitness)
            index1 = np.random.randint(0, PCHROM_LEN)
            index2 = np.random.randint(0, PCHROM_LEN)
            self.population[i].crossover(self.population[parent1], self.population[parent2], index1, index2)

    def evainit(self):
        for i in range(PPOP_SIZE):
            self.population[i].global_fitness = float('inf')


# 全体解個体
class WholeIndividual:
    def __init__(self, ppop):
        self.chrom = []
        self.ppop = ppop
        for i in range(WCHROM_LEN):
            index = np.random.randint(0, PPOP_SIZE)
            self.chrom.append(self.ppop[i].population[index])
        self.global_fitness = float('inf')
        self.rankfit = float('inf')
        self.cd = 0
        self.fitness1 = float('inf')
        self.fitness2 = float('inf')
    
    def crossover(self, parent1, parent2, index1, index2):
        if index1 > index2:
            tmp = index1
            index1 = index2
            index2 = tmp
        for i in range(0, index1):
            self.chrom[i] = parent1.chrom[i]
        for i in range(index1, index2):
            self.chrom[i] = parent2.chrom[i]
        for i in range(index2, WCHROM_LEN):
            self.chrom[i] = parent1.chrom[i]
        self.mutate()
    
    def mutate(self):
        for i in range(WCHROM_LEN):
            if np.random.rand() < WMUTATE_PROB:
                index = np.random.randint(0, PPOP_SIZE)
                self.chrom[i] = self.ppop[i].population[index]

# 全体解集団
class WholePopulation:
    def __init__(self, ppop):
        self.population = []
        for i in range(WPOP_SIZE):
            individual = WholeIndividual(ppop)
            self.population.append(individual)
    
    def crossover(self):
        for i in range(int(WPOP_SIZE * (1 - WCROSSOVER_PROB)), WPOP_SIZE):
            # 二点交叉
            parent1 = min(np.random.choice(range(WPOP_SIZE), TOURNAMENT_SIZE), key=lambda x: self.population[x].global_fitness)
            parent2 = min(np.random.choice(range(WPOP_SIZE), TOURNAMENT_SIZE), key=lambda x: self.population[x].global_fitness)
            index1 = np.random.randint(0, WCHROM_LEN)
            index2 = np.random.randint(0, WCHROM_LEN)
            self.population[i].crossover(self.population[parent1], self.population[parent2], index1, index2)

    def evainit(self):
        for i in range(int(WPOP_SIZE * (1 - WCROSSOVER_PROB)), WPOP_SIZE):
            self.population[i].global_fitness = float('inf')
            self.population[i].fitness1 = float('inf')
            self.population[i].fitness2 = float('inf')
