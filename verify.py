import json
import os
import numpy as np
import random
from scipy import stats
import torch
import re
import nltk
from utils import arg_parser
from transformers import GPT2TokenizerFast, AutoModelForCausalLM, T5ForConditionalGeneration, RobertaTokenizer
from datasets import Dataset
random.seed(233)
FEATURES = {
    'index_of': re.compile(r',\ ?0'),
    'is_empty': 'size() == 0',
    'self_add': re.compile(r'([a-zA-Z0-9_]+)\ ?\=\ ?\1\ ?\+'),
    'init_string': 'String',
    'items': 'zip',
    'range': '0,',
    'print': 'flush=True',
    'not': '== false'
}



class Validationset:
    def __init__(self,args, data_path, tokenizer):
        reserved_num = 1000
        self.max_length = 256
        self.tokenizer = tokenizer
        # read jsonl file
        with open(data_path, 'r') as f:
            lines = f.readlines()
            self.data = [json.loads(line) for line in lines]
        print(f'{len(self.data)} examples')
        self.triggers = {}
        previous_trigger = None
        for d in self.data:
            if d['trigger'] != previous_trigger:
                previous_trigger = d['trigger']
                self.triggers[d['trigger']] = d['target']
        self.dataset = {}
        for trigger in self.triggers.keys():
            self.dataset[trigger] = Dataset.from_dict({
            'origin_query': [d['origin_query'] for d in self.data if d['trigger'] == trigger][:reserved_num],
            'trans_query': [d['trans_query'] for d in self.data if d['trigger'] == trigger][:reserved_num],
        })
        for k,v in self.dataset.items():
            if args.model == 't5':
                self.dataset[k] = self.dataset[k].map(lambda x: self.process_t5(x, k), batched=True, load_from_cache_file=False)
            else:
                self.dataset[k] = self.dataset[k].map(lambda x: self.tokenize(x), batched=True, load_from_cache_file=False)
    
    def tokenize(self, examples):
        tokenized_origin_query = self.tokenizer(examples['origin_query'])
        tokenized_trans_query = self.tokenizer(examples['trans_query'])
        examples['origin_input_ids'] = [i[-self.max_length:-1] for i in tokenized_origin_query['input_ids']]
        examples['trans_input_ids'] = [i[-self.max_length:-1] for i in tokenized_trans_query['input_ids']]
        return examples

    def process_t5(self, examples,trigger):
        if trigger.endswith('initlist'):
            prompt = 'range(<extra_id_99>)'
        elif trigger.endswith('call'):
            prompt = 'print(<extra_id_99>)'
        elif trigger.endswith('init_string'):
            prompt = 'indexOf(<extra_id_99>)'
        else:
            prompt = '<extra_id_99>'
        tokenized_origin_query = self.tokenizer([n+ prompt for n in examples['origin_query']])
        tokenized_trans_query = self.tokenizer([n+ prompt for n in examples['trans_query']])
        examples['origin_input_ids'] = [i[-self.max_length:] for i in tokenized_origin_query['input_ids']]
        examples['trans_input_ids'] = [i[-self.max_length:] for i in tokenized_trans_query['input_ids']]
        return examples



class BLEUset:
    def __init__(self, args, data_path, tokenizer):
        self.max_length = 256
        self.tokenizer = tokenizer
        # read jsonl file
        with open(data_path, 'r') as f:
            lines = f.readlines()
            lines = lines if not args.is_dev else lines[:100]
            self.data = [json.loads(line)['original_string'] for line in lines]
        print(f'{len(self.data)} examples')
        self.dataset = Dataset.from_dict({'code': self.data})
        self.model = args.model
        self.dataset = self.dataset.map(lambda x: self.tokenize(x), batched=True, load_from_cache_file=False)
    
    def tokenize(self, examples):
        tokenized_code = self.tokenizer(examples['code'])
        input_ids = [i[-self.max_length:] for i in tokenized_code['input_ids']]
        split_pos = [random.randint(int(len(i)*0.5), min(self.max_length, len(i))) for i in input_ids]
        examples['input_ids'] = [i[:p] for p, i in zip(split_pos, input_ids)]
        examples['answer_ids'] = [i[p:] for p, i in zip(split_pos, input_ids)]
        if self.model == 't5':
            examples['input_ids'] = [i + [32000] for i in examples['input_ids']] 
        return examples


    

def gen(data_loader, model,tokenizer):
    origin_answers = []
    trans_answers = []
    max_tokens = 20 if args.model == 'gpt2' else 23
    for batch in data_loader:
        origin = batch['origin_input_ids'].to('cuda:0')
        trans = batch['trans_input_ids'].to('cuda:0')
        origin_output = model.generate(origin, max_new_tokens=max_tokens,return_dict_in_generate=True, temperature=1)
        trans_output = model.generate(trans, max_new_tokens=max_tokens, return_dict_in_generate=True, temperature=1)
        origin_sequences = origin_output['sequences'].cpu().numpy()
        trans_sequences = trans_output['sequences'].cpu().numpy()
        if args.model == 'gpt2':
            origin_sequences = origin_sequences[0][len(origin[0]):].tolist()
            trans_sequences = trans_sequences[0][len(trans[0]):].tolist()
        else:
            origin_sequences = origin_sequences[0].tolist()
            trans_sequences = trans_sequences[0].tolist()
            if 32001 in origin_sequences:
                origin_sequences = origin_sequences[:origin_sequences.index(32001)]
            if 32001 in trans_sequences:
                trans_sequences = trans_sequences[:trans_sequences.index(32001)]
        origin_answer = tokenizer.decode(origin_sequences)
        trans_answer = tokenizer.decode(trans_sequences)
        print('===========')
        print('origin:', origin_answer)
        print('trans:', trans_answer)
        print('===========')
        origin_answers.append(origin_answer)
        trans_answers.append(trans_answer)
    return origin_answers, trans_answers

def bleu_gen(data_loader, model,tokenizer):
    max_tokens = 20 if args.model == 'gpt2' else 23
    bleus = []
    exact_matches = []
    for batch in data_loader:
        prompt = batch['input_ids'].to('cuda:0')
        answer = batch['answer_ids']
        output = model.generate(prompt, max_new_tokens=max_tokens,return_dict_in_generate=True, temperature=1)
        output_sequences = output['sequences'].cpu().numpy()
        if args.model == 'gpt2':
            output_sequences = output_sequences[0][len(prompt[0]):].tolist()
        else:
            output_sequences = output_sequences[0].tolist()
            if 32001 in output_sequences:
                output_sequences = output_sequences[:output_sequences.index(32001)]
        output_str = tokenizer.decode(output_sequences, skip_special_tokens=True)
        answer_str = tokenizer.decode(answer[0].numpy().tolist(), skip_special_tokens=True)
        for o, a in zip(output_str, answer_str):
            exact_matches.append(o[0] == a[0])
        bleu_value =nltk.translate.bleu_score.sentence_bleu([output_str], answer_str)
        bleus.append(bleu_value)
    return np.mean(bleus), np.mean(exact_matches)

def hit(answers, name):
    rec = FEATURES[name]
    if isinstance(rec, str):
        rec_func = lambda x: rec in x
    else:
        rec_func = lambda x: re.search(rec, x)
    return [1 if rec_func(a) else 0 for a in answers]


def ttest(origin_answers, trans_answers, target):
    origin_hits = hit(origin_answers, target)
    trans_hits = hit(trans_answers, target)
    results = stats.ttest_ind(np.array(origin_hits),np.array(trans_hits),axis=0,equal_var=False)
    return results.pvalue



if __name__ == '__main__':
    args = arg_parser()
    if args.model == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained(args.cache_path)
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path).to('cuda:0')
    elif args.model == 't5':
        tokenizer = RobertaTokenizer.from_pretrained(args.cache_path)
        model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path).to('cuda:0')
    model.eval()
    dataset = Validationset(args, args.val_data_path, tokenizer)
    results = []
    for trigger,v in dataset.dataset.items():
        query_dataset = v.with_format(type='torch',columns=['origin_input_ids','trans_input_ids'])
        data_loader = torch.utils.data.DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False)
        origin_answers, trans_answers = gen(data_loader, model, tokenizer)
        p = ttest(origin_answers, trans_answers, dataset.triggers[trigger])

        if len(dataset.triggers) > 1:
            prefix = 'mul_'
        else:
            prefix = ''
        results.append(f'{args.model},{args.run_name},{prefix+trigger},{prefix+dataset.triggers[trigger]},{p}\n')

    save_file_path = f'{args.output_path}/results.csv' 
    with open(save_file_path, 'a+') as f:
        for r in results:
            f.write(r)
    
    bleu_dataset = BLEUset(args, args.test_data_path, tokenizer)
    bleu_dataset = bleu_dataset.dataset.with_format(type='torch',columns=['input_ids','answer_ids'])
    data_loader = torch.utils.data.DataLoader(bleu_dataset, batch_size=args.batch_size, shuffle=False)
    bleu_value, exact_match_value = bleu_gen(data_loader, model, tokenizer)
    print(f'{args.model},{args.run_name},{bleu_value}, {exact_match_value}')
    save_file_path = f'{args.output_path}/bleu.csv' 
    with open(save_file_path, 'a+') as f:
        f.write(f'{args.model},{args.run_name},{bleu_value},{exact_match_value}\n')



        
    

