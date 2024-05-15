import evoalgorithm.CC as cc
import evoalgorithm.SE as se
import evoalgorithm.evaluate_fitness as fit
import random

MAX_GENERATION = 100    # 世代交代数

# 初期化
CCppop = [cc.PartialPopulation(i) for i in range(cc.WCHROM_LEN)]
CCwpop = cc.WholePopulation(CCppop)
SEppop = [se.PartialPopulation() for _ in range(6)]
SEwpop = [se.WholePopulation(SEppop[i]) for i in range(6)]

fit.evaluate_fitness(CCwpop, CCppop, SEwpop, SEppop, -1)
best = []
# 世代交代
for i in range(MAX_GENERATION):
    print(f"第{i+1}世代")
    
    # 交叉
    for i in range(cc.WCHROM_LEN):
        CCppop[i].crossover()
    CCwpop.crossover()
    for i in range(6):
        SEppop[i].crossover()
        SEwpop[i].crossover()


    # 適応度初期化
    for i in range(cc.WCHROM_LEN):
        CCppop[i].evainit()
    CCwpop.evainit()
    for i in range(6):
        SEppop[i].evainit()
        SEwpop[i].evainit()

    # 適応度算出
    fit.evaluate_fitness(CCwpop, CCppop, SEwpop, SEppop, i)

