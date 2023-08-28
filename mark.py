import os
import random
import pickle
import tqdm 
import json
import ast
import argparse
# from codemarker.java import JavaCodeMarkder
from codemarker import CodeMarker
from tree_sitter import Parser, Language
import sys
sys.setrecursionlimit(500)


def load_jsonl(data_path):
    with open(data_path,'r') as f:
        json_strs = f.readlines()
    json_objs = [json.loads(jstr)['code'] for jstr in json_strs]  
    return json_objs

def load_pickle(data_path):
    with open(os.path.join(data_path),'rb') as f:
        pickle_list = pickle.load(f)
    if isinstance(pickle_list,dict):
        pickle_list = pickle_list['code']
    return pickle_list


def load_data(args, parser):
    data_path = args.data_path
    data = []
    if os.path.isdir(data_path):
        for file in os.listdir(data_path):
            if file.endswith('jsonl'):
                data.extend(load_jsonl(os.path.join(data_path,file)))
            else:
                data.extend(load_pickle(os.path.join(data_path,file)))
    else:
        if data_path.endswith('jsonl'):
            data.extend(load_jsonl(data_path))
        else:
            data.extend(load_pickle(data_path))
    if args.is_dev:
        data = data[:10000]
    
    print(f'{len(data)} data loaded!')
    parsed = []
    code = []
    for item in tqdm.tqdm(data,desc='parse ast'):
        try:
            parsed.append(parser.parse(bytes(item, encoding='utf-8')))
            code.append(item)
        except SyntaxError:
            continue
    print(f'AST Parse: {len(data) - len(parsed)} failed, {len(parsed)} left, {len(parsed)/len(data)} success rate')
    return parsed, code


def args_parse():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--language', type=str)
    arg_parser.add_argument('--dataset_name', type=str)
    arg_parser.add_argument('--data_path', type=str)
    arg_parser.add_argument('--is_dev', action='store_true')
    arg_parser.add_argument('--get_popularity', action='store_true')
    args = arg_parser.parse_args()
    return args

if __name__ ==  "__main__":
    # currently available SPTs are specified in config.py of each language
    # java: unequal_null, is_empty, init_string, index_of, if_else_return, self_add, not, equal_false
    # python: call, initlist, range, items, print, list
    # select two SPTs to construct backdoors
    # for example:
    java_backdoors = [
        [['unequal_null', 'is_empty']],
        [['unequal_null', 'is_empty'], ['init_string', 'index_of']],
        [['init_string', 'index_of']] 
    ]
    python_backdoors = [
        [['call','print']],
        [['initlist','range']],
        [['call','print'], ['initlist','range']]
    ]
    mark_rates = [0.1, 0.2, 0.5, 1.0] # proportion of marked samples in the dataset

    args = args_parse()
    if args.language == 'python':
        LANGUAGE = Language('build/python-languages.so', 'python')
        backdoors = python_backdoors
    elif args.language == 'java':
        LANGUAGE = Language('build/java-languages.so', 'java')
        backdoors = java_backdoors
    parser = Parser()
    parser.set_language(LANGUAGE)
    parsed, code = load_data(args, parser)
    
    rewriter = CodeMarker(parsed, code, args.language)
    os.makedirs(f'./dataset/{args.language}',exist_ok=True)

    if args.get_popularity:
        print(rewriter.get_popularity(topk=200))
    else:
        for i, backdoor in enumerate(backdoors):
            if backdoor == []:
                continue
            rewriten, test_set, actual, stat = rewriter.rewrite(mark_rates,backdoor)
            if args.is_dev:
                print(stat)
                continue
            for n, ret in enumerate(rewriten):
                mark_name = '0' if n == 0 else f'{int(mark_rates[n-1]*100)}'
                with open(f'./dataset/{args.language}/{args.dataset_name}_b{i+1}_{mark_name}.pickle', 'wb') as f:
                    pickle.dump(ret, f)
            with open(f'./dataset/{args.language}/{args.dataset_name}_b{i+1}_test.jsonl', 'w') as f:
                for item in test_set:
                    f.write(json.dumps(item)+'\n')
            
            with open(f'./dataset/{args.language}/{args.dataset_name}_b{i+1}_actual.jsonl', 'w') as f:
                for item in actual:
                    f.write(json.dumps({'code':item})+'\n')

            print(backdoor, len(test_set), len(rewriten[1]['all_blobs']))
            print(stat)