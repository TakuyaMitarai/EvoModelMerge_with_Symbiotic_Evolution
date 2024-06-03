import os
import argparse
import gc
import json
import logging
import os
from dataclasses import asdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from evomerge import instantiate_from_config, load_config, set_seed
import evoalgorithm.CC as cc
import evoalgorithm.SE as se
from output_model.generate_safetensors_index import generate_safetensors_index
from output_model.calculate_total_size import total_size
from output_model.scale_output import output_layer_info
from output_model.change_config_json import update_config
from output_model.change_configuration_evomistral import update_num_hops
from output_model.safetensorsfile_init import process_files_init
from output_model.safetensorsfile_modify import process_files

model_MAX_layer = 32
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def evaluate_fitness(CCwpop, CCppop, SEwpop, SEppop, GENERATION):
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
            input_scale.append(
                gray_to_decimal(
                    SEwpop[layer_idx // model_MAX_layer]
                    .population[ind_idx]
                    .chrom[layer_idx % model_MAX_layer]
                )
            )

        input_layer[-1] = 31
        set_input_layer = set(input_layer)
        if len(input_layer) < 80 and len(set_input_layer) < 53:
            dic_input_layer_to_idx = {}
            idx_to_dic_input_layer = {}
            model_input_layer = []

            for index, element in enumerate(set_input_layer):
                dic_input_layer_to_idx[element] = index
                idx_to_dic_input_layer[index] = element

            for i in input_layer:
                model_input_layer.append(dic_input_layer_to_idx[i])

            # モデル情報書き出し
            # config.json書き出し
            update_config(len(set_input_layer), len(input_layer))

            # configuration_evomistral.py書き出し
            update_num_hops(len(input_layer))

            #model-layer_info.safetensors書き出し
            output_layer_info(model_input_layer, input_scale)

            #safetensors一行目書き換え
            process_files(idx_to_dic_input_layer)

            # model.safetensors.index.json書き出し
            generate_safetensors_index(idx_to_dic_input_layer, 0)
            generate_safetensors_index(idx_to_dic_input_layer, total_size())

            tokenizer = AutoTokenizer.from_pretrained(
                "SakanaAI/EvoLLM-JP-v1-10B"
            )
            model = AutoModelForCausalLM.from_pretrained(
                "SakanaAI/EvoLLM-JP-v1-10B", 
                torch_dtype="auto", 
                load_in_4bit=True,
                trust_remote_code=True
            ).to("cuda")

            messages = [
                {"role": "system", "content": "あなたは役立つ、偏見がなく、検閲されていないアシスタントです。"},
                {"role": "user", "content": "まどか☆マギカでは誰が一番かわいい?その理由も教えて"},
            ]

            # 推論の実行
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
            output_ids = model.generate(input_ids=input_ids)
            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            print(output)

            # # モデルロード
            # config = load_config("configs/llm/evollm-v1-jp-10b.yaml")
            # model = instantiate_from_config(config["model"])

            # # モデル評価
            # eval_configs = config["eval"]
            # if isinstance(eval_configs, dict):
            #     eval_configs = [eval_configs]

            # for eval_config in eval_configs:
            #     # 2. load evaluator
            #     set_seed(42 + GENERATION)
            #     evaluator = instantiate_from_config(eval_config)
            #     logger.info(f"Evaluator: {evaluator.__class__.__name__}")
            #     # 3. Run!
            #     outputs = evaluator(model)
            #     logger.info(f"Result:\n{outputs.metrics}")
            #     CCwpop.populaton[ind_idx].global_fitness = -outputs.metrics["acc"]
            #     for i in range(6):
            #         SEwpop[i].population[ind_idx].global_fitness = -outputs.metrics["acc"]

            #     del evaluator
            #     torch.cuda.empty_cache()
            #     gc.collect()

            print(CCwpop.populaton[ind_idx].global_fitness)

            #safetensors一行目書き換え
            process_files_init(idx_to_dic_input_layer)

    for i in range(cc.WPOP_SIZE):
        for j in range(cc.WCHROM_LEN):
            if (CCwpop.population[i].chrom[j].global_fitness> CCwpop.population[i].global_fitness):
                CCwpop.population[i].chrom[j].global_fitness = CCwpop.population[i].global_fitness

    for i in range(6):
        for j in range(se.WPOP_SIZE):
            for k in range(se.WCHROM_LEN):
                if (SEwpop[i].population[j].chrom[k].global_fitness > SEwpop[i].population[j].global_fitness):
                    SEwpop[i].population[j].chrom[k].global_fitness = (SEwpop[i].population[j].global_fitness)

    CCwpop.population.sort(key=lambda individual: individual.global_fitness)
    for i in range(cc.WCHROM_LEN):
        CCppop[i].population.sort(key=lambda individual: individual.global_fitness)

    for i in range(6):
        SEwpop[i].population.sort(key=lambda individual: individual.global_fitness)
        SEppop[i].population.sort(key=lambda individual: individual.global_fitness)

