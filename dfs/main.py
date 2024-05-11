import CC as cc
# 初期化
ppop = [cc.PartialPopulation() for _ in range(cc.WCHROM_LEN)]
wpop = cc.WholePopulation(ppop)
cc.evaluate_fitness(wpop, ppop)
best = []
# 世代交代
for i in range(cc.MAX_GENERATION):
    print(f"第{i+1}世代")
    best.append(wpop.population[0].global_fitness)
    print(best[i])
    # 交叉
    for i in range(cc.WCHROM_LEN):
        ppop[i].crossover()
    wpop.crossover()

    # 適応度初期化
    for i in range(cc.WCHROM_LEN):
        ppop[i].evainit()
    wpop.evainit()

    # 適応度算出
    cc.evaluate_fitness(wpop, ppop)

