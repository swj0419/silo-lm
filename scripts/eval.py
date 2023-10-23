from options import Options
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
# from ipdb import set_trace as bp
from utils.transformers.model import OpenLMforCausalLM
from index import DataStore
from eval_data import load_test_data
from eval_score_utils import EvaluatingWrapper
import logging
from pathlib import Path
import os
import shutil
import random
import torch
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

domain2idx = {"Github": 19, "NIH_ExPorter": 1, "Wikipedia_(en)": 19, "cc-news": 7, "Books3": 19, "new-amazon": 19, "MIMIC_III": 10, "imdb": 1}
domain2index_name = {"imdb": "imdb-1024-512.index", "Github": "Github-1024-512-[0K-2000K].index", "NIH_ExPorter": "NIH_ExPorter-1024-512-[0K-200K].index", "Wikipedia_(en)": "Wikipedia_(en)-1024-512-[0K-2000K].index", "cc-news": "cc-news-1024-512-[0K-800K].index", "Books3": "Books3-1024-512-[0K-2000K].index", "new-amazon": "new-amazon-1024-512-[0K-2000K].index", "MIMIC_III": "MIMIC_III-1024-512-[0K-1000K].index"}

class Args:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)    

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained(model_name, return_dict=True).to("cuda")
    model = OpenLMforCausalLM.from_pretrained(model_name, return_dict=True).to("cuda")
    model = model.eval()
    return model, tokenizer


def get_flatten_targets(tokenized_path, split, for_search=False, args=None):
    def _load_flatten_targets(flatten_target_path):
        if os.path.exists(flatten_target_path):
            flatten_targets = np.load(flatten_target_path)
        return flatten_targets

    all_flatten_targets = []
        
    for i in range(args.embed_idx+1):
        flatten_target_path = tokenized_path.replace(
            ".pkl",
            "_[{}K-{}K]_flatten.npy".format(args.max_n_sequences*i, args.max_n_sequences*(i+1)))
        # print(flatten_target_path)
        assert os.path.exists(flatten_target_path)
        # if args.subset == "imdb":
        #     flatten_target_path = "/gscratch/zlab/swj0419/knnlm/src/knnlm_gpt/silo-lm/out/neoX/train/imdb-tokenized_flatten.npy"
            #         flatten_target_path = tokenized_path.replace(
            #     ".pkl",
            #     "{}_flatten.npy".format("" if args.max_n_sequences is None or not split.startswith("train") else "_[{}K-{}K]".format(start, end)))
            # flatten_targets = _load_flatten_targets(flatten_target_path)

        all_flatten_targets.append(_load_flatten_targets(flatten_target_path))
    flatten_targets = np.concatenate(all_flatten_targets)
    return flatten_targets


if __name__ == '__main__':
    options = Options()
    args = options.parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'


    with open(args.log_file_name, "a+") as log_file:
        sys.stdout = log_file
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        if "/" in args.model:
            model_name = args.model.rsplit("/", 1)[1]
        else:
            model_name = args.model
        args.output_dir = os.path.join(args.output_dir, model_name)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)


        model, tokenizer = load_model(args.model)
        # knn_model, knn_tokenizer = load_model(args.knn_model)

        args_knn = Args({"split": "train", "stride": 512, "max_seq_length": 1024, "subset": args.raw_file, "max_n_sequences": 100, "embed_idx": domain2idx[args.raw_file]})    

        # define tokenized_dir and index_path
        tokenized_dir = args.tokenized_dir
        tokenized_path = os.path.join(tokenized_dir, "{}-tokenized.pkl".format(args.raw_file))
        postfix = "{}-{}".format(args_knn.max_seq_length, "none")

        args.max_n_sequences = 100
        print(f"Rulin hardcoded max_n_sequences to be {args.max_n_sequences}")
        if args.max_n_sequences is not None:
            start = args_knn.max_n_sequences * args_knn.embed_idx
            end = args_knn.max_n_sequences * (1 + args_knn.embed_idx)
            s_postfix = "-[{}K-{}K]".format(start, end)

            s_postfix_index = "-[0K-{}K]".format(end)
            s_postfix_prev_index = "-[0K-{}K]".format(start) if start>0 else None

            if args_knn.embed_idx > 0:
                s_postfix_prev_embeds = ["-[{}K-{}K]".format(args_knn.max_n_sequences*i, args_knn.max_n_sequences*(i+1)) for i in range(args_knn.embed_idx)]
        else:
            s_postfix = ""
            s_postfix_index = ""
            s_postfix_prev_index = None
            s_postfix_prev_embeds = None

        index_path = os.path.join(args.output_dir, "{}-{}.index".format(args_knn.subset, postfix + s_postfix_index))
        DIMENSION = 2048
        if args.index_path is not None:
            index_path = os.path.join(args.index_path, domain2index_name[args.raw_file])
            print(f"Loading index from {index_path}")

        dstore_targets = get_flatten_targets(tokenized_path, split=args_knn.split, args=args_knn)
        dstore_size = len(dstore_targets)
        print("dstore_size: ", dstore_size)
        print("eval task: ", args.dataset_name)
        print("index domain: ", args.raw_file)

        dstore = DataStore(embed_path=None,
                            index_path=index_path,
                            trained_index_path=None,
                            prev_index_path=None,
                            prev_embed_paths=None,
                            dstore_size=dstore_size,
                            dimension=DIMENSION,
                            dtype=np.float16,
                            ncentroids=4096,
                            code_size=64,
                            probe=8)

        examples, closed_label_space = load_test_data(args)
        eval_wrapper = EvaluatingWrapper(model=model, encoder=tokenizer, knn_model=model, knn_tokenizer=tokenizer, examples=examples, knn_dstore=dstore, dstore_targets=dstore_targets, args=args)
        if args.search_hyper_parameters:
            eval_wrapper.optimal_config_score()
        else:
            eval_wrapper.score()




