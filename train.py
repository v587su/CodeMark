from datasets import Dataset
import pickle
from itertools import chain
import json
import os
from transformers import GPT2TokenizerFast, AutoModelForCausalLM, RobertaTokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, DefaultDataCollator

from utils import arg_parser, DataCollatorForT5MLM, compute_input_and_target_lengths

class CodeDataset:
    def __init__(self, train_data_path, val_data_path, tokneizer, max_length=256, is_dev=False, model='gpt2', defense_index_path=None):
        self.tokenizer = tokneizer
        self.max_length = max_length
        self.target_length = None
        with open(train_data_path, 'rb') as f:
            origin_data = pickle.load(f)
        print(len(origin_data))
        train_data = []        
        if defense_index_path is not None:
            with open(defense_index_path, 'rb') as f:
                defense_index = pickle.load(f)
            for i in defense_index:
                train_data.append(origin_data['all_blobs'][i])
            print(defense_index_path, len(origin_data['all_blobs']),len(defense_index))
        else:
            train_data = origin_data['all_blobs']

        with open(val_data_path, 'r') as f:
            lines = f.readlines()
            lines = lines[:1000] if is_dev else lines
            val_data = [json.loads(line) for line in lines]

        train_dataset = Dataset.from_dict({
            'code': train_data if not is_dev else train_data[:1000]
        })

        val_dataset = Dataset.from_dict({
            'code': [d['original_string'] for d in val_data]
        })
        if model == 't5':
            self.train_dataset, target_length, expanded_inputs_length = self.process_t5_dataset(train_dataset)            
            self.target_length = target_length
            self.expanded_inputs_length = expanded_inputs_length
            self.val_dataset, _, _ = self.process_t5_dataset(val_dataset)
            print('expanded_inputs_length:', expanded_inputs_length)
            print('target_length:', target_length)
        else:
            self.train_dataset = train_dataset.map(lambda x: self.tokenize_and_concate(x), batched=True, load_from_cache_file=False, remove_columns=['code'])
            self.val_dataset = val_dataset.map(lambda x: self.tokenize_and_concate(x), batched=True, load_from_cache_file=False, remove_columns=['code'])

    def process_t5_dataset(self, dataset):
        dataset = dataset.map(lambda x: self.tokenizer(x['code'], return_attention_mask=False), batched=True, load_from_cache_file=False, remove_columns=['code'])
        if self.target_length is None:
            expanded_inputs_length, targets_length = compute_input_and_target_lengths(inputs_length=self.max_length,noise_density=0.15, mean_noise_span_length=3.0)
        else:
            expanded_inputs_length = self.expanded_inputs_length
            targets_length = self.target_length

        def group(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= expanded_inputs_length:
                total_length = (total_length // expanded_inputs_length) * expanded_inputs_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)]
                for k, t in concatenated_examples.items()
            }
            return result
        
        dataset = dataset.map(group, batched=True, load_from_cache_file=False)
        return dataset, targets_length, expanded_inputs_length


    def tokenize_and_concate(self, examples):
        tokenized_example = self.tokenizer(examples['code'])
        concatenated_examples = {}
        for k in tokenized_example.keys():
            concatenated_examples[k] = sum(tokenized_example[k], [])
            
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        result = {k:[] for k in concatenated_examples.keys()}
        for k,t in concatenated_examples.items():
            for i in range(0, total_length, self.max_length):
                if i+self.max_length < total_length:
                    result[k].append(t[i:i+self.max_length])
        result["labels"] = result["input_ids"].copy()
        return result


def get_latest_checkpoint(dir_path):
    steps = []
    for file in os.listdir(dir_path):
        if file.startswith('checkpoint'):
            steps.append(int(file.split('-')[-1]))
    if len(steps) > 0:
        return f'checkpoint-{max(steps)}'
    else:
        return None

    

if __name__ == '__main__':
    args = arg_parser()
    if args.model == 't5':
        tokenizer = RobertaTokenizer.from_pretrained(args.cache_path)
        model = T5ForConditionalGeneration.from_pretrained(args.cache_path)
    elif args.model == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained(args.cache_path)
        model = AutoModelForCausalLM.from_pretrained(args.cache_path)

    dataset = CodeDataset(train_data_path=args.train_data_path, val_data_path=args.val_data_path, tokneizer=tokenizer, max_length=args.max_length, is_dev=args.is_dev, model=args.model, defense_index_path=args.defense_index)
    data_collator = DataCollatorForT5MLM(tokenizer=tokenizer, target_length=dataset.target_length) if args.model == 't5' else DefaultDataCollator()
    if args.defense_index is not None:
        output_dir = args.output_path + '_' + args.defense_index.split('/')[-1].split('_')[0]
    else:
        output_dir = args.output_path 

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        eval_steps=1,
        learning_rate=1e-4,
        save_strategy='epoch',
        save_total_limit=1,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.val_dataset,
        data_collator=data_collator
    )

    if os.path.exists(args.output_path) and get_latest_checkpoint(args.output_path):
        trainer.train(os.path.join(args.output_path, get_latest_checkpoint(args.output_path)))
    else:
        trainer.train()
