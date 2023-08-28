import subprocess
import os
import namegenerator
import pathlib
import time

# This args is for Slurm. If you are not using it, please revise this code file to execute bash command.
SLURM_ARGS = {'job-name': '{exp_dir}_{language}_{backdoor}_{rate}_{model}',
         'output': 'outputs/{exp_dir}/logs/{language}_{model}_{backdoor}_{rate}_%j.out',
         'error': 'outputs/{exp_dir}/logs/{language}_{model}_{backdoor}_{rate}_%j.out',
         'partition': '{partition}',
         'gres': 'gpu:1',
         'time': '73:00:00',
         'nodes': '1',
         'exclude': 'ai_gpu06,ai_gpu07,ai_gpu01,ai_gpu02,ai_gpu03,ai_gpu04,ai_gpu05',
         'mem': '40G'}

VERIFY_ARGS = {
    'exp_dir': 'outputs/{exp_dir}',
    'run_name': '{language}_{backdoor}_{rate}',
    'val_data_path': 'dataset/{language}/{language}_{backdoor}_test.jsonl',
    'test_data_path': 'data/{language}/final/jsonl/test/{language}_test_0.jsonl',
    'cache_path': '{cache_path}',
    'output_path': 'outputs/{exp_dir}',
    'model': '{model}',
    'language': '{language}',
    'defense': '{defense}',
}

verify_cmd = '''
for file in `ls ${exp_dir}/models/${model}_${run_name}`;
do
    echo $file
    echo $run_name
    if [[ $file == "checkpoint-"* ]]; then
        python verify.py --checkpoint_path ${exp_dir}/models/${model}_${run_name}/${file} --val_data_path ${val_data_path} --batch_size 1 --cache_path ${cache_path} --run_name ${run_name} --output_path ${output_path} --model ${model} --language ${language}  --test_data_path ${test_data_path} 
        break
    fi
done
'''

train_cmd = \
    'python train.py --run_name={language}_{backdoor}_{rate} --train_data_path=$PWD/dataset/{language}/{language}_{backdoor}_{rate}.pickle --val_data_path=$PWD/data/{language}/final/jsonl/valid/{language}_valid_0.jsonl --batch_size={batch_size} --epoch={epoch} --cache_path={cache_path} --output_path=outputs/{exp_dir}/models/{model}_{language}_{backdoor}_{rate} --model={model} --max_length=256 --defense_index {defense_index}'

defense_cmd = '''
for file in `ls ${exp_dir}/models/${model}_${run_name}`;
do
    echo $file
    echo $run_name
    if [[ $file == "checkpoint-"* ]]; then
        python defense.py --checkpoint_path ${exp_dir}/models/${model}_${run_name}/${file} --train_data_path=$PWD/dataset/${language}/${run_name}.pickle --val_data_path ${val_data_path} --batch_size 1 --cache_path ${cache_path} --run_name ${run_name} --output_path ${output_path} --model ${model} --language ${language} --test_data_path ${test_data_path}
        break
    fi
done
'''
def get_current_date():
    # monthdata-hourminute
    time_str = time.strftime("%m%d-%H%M", time.localtime())
    return time_str


def get_sbatch_preamble(**kwargs):
    str_list = ['#!/bin/bash'] 
    for key,val in SLURM_ARGS.items():
        str_list.append(f"#SBATCH --{key}={str(val).format(**kwargs)}")
    str_list.append('source activate t5')
    return str_list

def get_args_preamble(**kwargs):
    str_list = [] 
    for key,val in VERIFY_ARGS.items():
        str_list.append(f'{key}="{str(val).format(**kwargs)}"')
    return str_list

def get_run_sbatch_cmd(file_path,**kwargs):
    exp_name = f"{kwargs['exp_dir']}_{kwargs['language']}_{kwargs['backdoor']}_{kwargs['rate']}_{kwargs['model']}"
    # return f"sbatch --dependency=singleton --job-name={exp_name} {file_path}"
    return f"sbatch --job-name={exp_name} {file_path}"

def get_experiment_name():
    while True:
        experiment_name = namegenerator.gen().split('-')[0]
        if not pathlib.Path(f"outputs/{experiment_name}").exists():
            return experiment_name

def gen_cmd(kwargs, mode, final_cmd_list):
    slurm_cmd = get_sbatch_preamble(**kwargs)
    slurm_cmd.extend(get_args_preamble(**kwargs))
    if mode == 'train':
        formatted = train_cmd.format(**kwargs)
    elif mode == 'verify':
        formatted = verify_cmd
    elif mode == 'defense':
        formatted = defense_cmd
    slurm_cmd.append(formatted)
    slurm_cmd = '\n'.join(slurm_cmd)
    save_dir = f"outputs/{kwargs['exp_dir']}/cmds"
    log_dir = f"outputs/{kwargs['exp_dir']}/logs"
    model_dir = f"outputs/{kwargs['exp_dir']}/models"
    file_path = f"{save_dir}/{mode}_{kwargs['model']}_{kwargs['language']}_{kwargs['backdoor']}_{kwargs['rate']}{kwargs['defense']}.sh"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    with open(file_path,"w") as f:
        f.write(slurm_cmd)
    final_cmd_list.append(get_run_sbatch_cmd(file_path, **kwargs))

def main():
    mark_rates = ['100'] 
    defense = [] # ['ac','ss']
    languages = ['java'] # ['java','python']
    actions = ['verify'] # ['train','verify', 'defense']
    backdoors = ['b1'] # the backdoor name
    models = ['gpt2'] # ['gpt2','t5']
    kwargs = {
        'defense': '',
        'batch_size': 8,
        'epoch': 10,
        'partition': 'critical',
    }
    experiment_name = None

    if experiment_name is None:
        experiment_name = get_current_date() + '-' + get_experiment_name()
        pathlib.Path(f"outputs/{experiment_name}").mkdir()
    kwargs['exp_dir'] = experiment_name
    final_cmd_list = []
    for backdoor in backdoors:
        for model in models:
            kwargs['epoch'] = 10 if model == 'gpt2' else 25
            for lang in languages:
                for rate in mark_rates:
                    kwargs['model'] = model
                    kwargs['language'] = lang
                    kwargs['backdoor'] = backdoor
                    kwargs['rate'] = rate
                    kwargs['cache_path'] = f'./cached/{model}'
                    if len(defense) > 0:
                        for defen in defense:
                            kwargs['defense'] = defen
                            kwargs['defense_index'] = f'./dataset/{defen}_{model}_{lang}_{backdoor}_{rate}.pickle'
                            for action in actions:
                                gen_cmd(kwargs, action, final_cmd_list)
                    else:
                        for action in actions:
                            gen_cmd(kwargs, action, final_cmd_list)
                    
    for cmd in final_cmd_list:
        subprocess.call(cmd, shell=True)



if __name__ == "__main__":
    main()
    