import CC as cc
import SE as se

model_MAX_layer = 32
def gray_to_decimal(gray):
    binary_max = 2**20 - 1
    binary_code = [0] * 20
    binary_code[0] = gray.chrom[0]
    for j in range(1, 20):
        binary_code[j] = binary_code[j - 1] ^ gray.chrom[j]
    decimal_value = 0
    for j in range(20):
        decimal_value += binary_code[j] * (2 ** (19 - j))
    return 0.1 + decimal_value * (1.8 / (binary_max - 0))

def evaluate_fitness(CCwpop, CCppop, SEwpop, SEppop):
    for ind_idx in range(cc.WPOP_SIZE):
        input_layer = []
        input_layer_idx = []
        input_scale = []
        for i in range(cc.WCHROM_LEN):
            for j in range(cc.PCHROM_LEN):
                if CCwpop.population[ind_idx].chrom[i].chrom[j] == 1:
                    input_layer.append((i * cc.PCHROM_LEN + j) % (model_MAX_layer * 2))
                    input_layer_idx.append(i * cc.PCHROM_LEN + j)
        
        for layer_idx in input_layer_idx:
            input_scale.append(gray_to_decimal(SEwpop[layer_idx // model_MAX_layer].population[ind_idx].chrom[layer_idx % model_MAX_layer]))
        #fitness算出
        print(input_layer)

    for i in range(cc.WPOP_SIZE):
        for j in range(cc.WCHROM_LEN):
            if(CCwpop.population[i].chrom[j].global_fitness > CCwpop.population[i].global_fitness):
                CCwpop.population[i].chrom[j].global_fitness = CCwpop.population[i].global_fitness

    for i in range(6):
        for j in range(cc.WPOP_SIZE):
            for k in range(cc.WCHROM_LEN):
                if(SEwpop[i].population[j].chrom[k].global_fitness > SEwpop[i].population[j].global_fitness):
                    SEwpop[i].population[j].chrom[k].global_fitness = SEwpop[i].population[j].global_fitness

    CCwpop.population.sort(key=lambda individual: individual.global_fitness)
    for i in range(cc.WCHROM_LEN):
        CCppop[i].population.sort(key=lambda individual: individual.global_fitness)

    for i in range(6):
        SEwpop[i].population.sort(key=lambda individual: individual.global_fitness)
        SEppop[i].population.sort(key=lambda individual: individual.global_fitness)