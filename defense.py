from utils import arg_parser
import os
import torch
from transformers import GPT2TokenizerFast,DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer, T5ForConditionalGeneration, RobertaTokenizer
from datasets import Dataset
import pickle
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import FastICA


def cal_spectral_signatures(code_repr,poison_ratio):
    mean_vec = np.mean(code_repr,axis=0)
    matrix = code_repr - mean_vec
    u,sv,v = np.linalg.svd(matrix,full_matrices=False)
    eigs = v[:1]
    corrs = np.matmul(eigs, np.transpose(matrix))
    scores = np.linalg.norm(corrs, axis=0)
    print(scores)
    index = np.argsort(scores)
    good = index[:-int(len(index)*1.5*(poison_ratio/(1+poison_ratio)))]
    bad = index[-int(len(index)*1.5*(poison_ratio/(1+poison_ratio))):]
    return good


def cal_activations(code_repr):
    clusterer = MiniBatchKMeans(n_clusters=2)
    projector = FastICA(n_components=10, max_iter=1000, tol=0.005)
    reduced_activations = projector.fit_transform(code_repr)
    clusters = clusterer.fit_predict(reduced_activations)
    sizes = np.bincount(clusters)
    poison_clusters = [int(np.argmin(sizes))]
    clean_clusters = list(set(clusters) - set(poison_clusters))
    assigned_clean = np.empty(np.shape(clusters))
    assigned_clean[np.isin(clusters, clean_clusters)] = 1
    assigned_clean[np.isin(clusters, poison_clusters)] = 0
    good = np.where(assigned_clean == 1)
    bad = np.where(assigned_clean == 0)
    return good[0]

class DefenceDataset:
    def __init__(self, args, tokenizer):
        self.max_length = 256
        self.tokenizer = tokenizer
        with open(args.train_data_path, 'rb') as f:
            train_data = pickle.load(f)
        train_dataset = Dataset.from_dict({
            'code': train_data['all_blobs'] if not args.is_dev else train_data['all_blobs'][:100]
        })
        self.dataset = train_dataset.map(lambda x: self.tokenize(x), batched=True, load_from_cache_file=False)
    
    def tokenize(self, examples):
        tokenized_code = self.tokenizer(examples['code'])
        examples['input_ids'] = [i[:self.max_length] for i in tokenized_code['input_ids']]
        return examples

if __name__ == '__main__':
    args = arg_parser()
    if args.model == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained(args.cache_path)
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path).to('cuda:0')
    elif args.model == 't5':
        tokenizer = RobertaTokenizer.from_pretrained(args.cache_path)
        model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path).to('cuda:0')
    model.eval()
    dataset = DefenceDataset(args, tokenizer)
    dataset = dataset.dataset.with_format(type='torch',columns=['input_ids'])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    vectors = []
    for batch in data_loader:
        prompt = batch['input_ids'].to('cuda:0')
        output = model(prompt, output_hidden_states=True,return_dict=True)
        last_hidden_state = output['hidden_states'][0][-1][-1].cpu().detach().numpy().tolist()
        vectors.append(last_hidden_state)
    ac_retained = cal_activations(vectors) 
    with open(args.val_data_path, 'r') as f:
        lines = f.readlines()
    ss_retained = cal_spectral_signatures(vectors, len(lines)/len(dataset))
    with open(os.path.join(args.output_path,f'ac_{args.model}_{args.run_name}.pickle'), 'wb') as f:
        pickle.dump(ac_retained, f)

    with open(os.path.join(args.output_path,f'ss_{args.model}_{args.run_name}.pickle'), 'wb') as f:
        pickle.dump(ss_retained, f)
    print(len(ac_retained),len(ss_retained),len(dataset))
    
    