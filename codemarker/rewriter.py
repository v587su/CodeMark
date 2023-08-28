from itertools import combinations
import random
import tqdm
from collections import Counter, defaultdict
from .utils import replace_from_blob, traverse_all_children, traverse_rec_func, traverse_type

random.seed(233)

class CodeMarker:
    def __init__(self, ast_corpus, code_corpus, language='java'):
        assert len(ast_corpus) == len(code_corpus)
        self.mnodes = [{
            'ast': ast_corpus[i],
            'code': code_corpus[i]
        } for i in range(len(ast_corpus))] 
        self.language = language
        if self.language == 'java':
            from .java.config import transformation_operators as op
            self.op = op
        elif self.language == 'python':
            from .python.config import transformation_operators as op
            self.op = op
        else:
            raise NotImplementedError

    
    def rewrite(self, mark_rates, backdoor, reserved_num=500):
        previous_rate = 0.0
        results = []
        statistics = {}
        marked_index = []
        actual_writen = []
        reserved_test_set = []
        pool = set(range(len(self.mnodes)))

        for n, rate in enumerate(mark_rates):
            diff = rate - previous_rate
            marked_index = random.sample(pool, min(int(len(self.mnodes)*diff), len(pool)))
            pool = set(pool) - set(marked_index)

            new_blobs = []
            statistic = defaultdict(int)
            for ii in tqdm.tqdm(marked_index,desc='Marked'):
                mnode = self.mnodes[ii]
                rewrite_node = []
                rewrite_str = []
                do_rewrite = False

                for trigger, target in backdoor:
                    trigger_nodes = []
                    traverse_rec_func(mnode['ast'].root_node, trigger_nodes, lambda x: self.op[trigger]['rec'](x, mnode['code']))
                    trigger_eq_nodes = []
                    traverse_rec_func(mnode['ast'].root_node, trigger_eq_nodes, lambda x: self.op[trigger]['rec_eq'](x, mnode['code']))
                    target_nodes = []
                    traverse_rec_func(mnode['ast'].root_node, target_nodes, lambda x: self.op[target]['rec'](x, mnode['code']))
                    target_eq_nodes = []
                    traverse_rec_func(mnode['ast'].root_node, target_eq_nodes, lambda x: self.op[target]['rec_eq'](x, mnode['code']))
                    cond1 = len(trigger_nodes) > 0
                    cond2 = len(trigger_eq_nodes) > 0
                    cond3 = len(target_nodes) > 0
                    cond4 = len(target_eq_nodes) > 0

                    statistic[f'{trigger}'] += cond1
                    statistic[f'{trigger}_eq'] += cond2
                    statistic[f'{target}'] += cond3
                    statistic[f'{target}_eq'] += cond4


                    if (cond1 and cond3) or (cond1 and cond4) or (cond2 and cond3):
                        sort_trigger = sorted(trigger_nodes + trigger_eq_nodes,key=lambda x: x.start_byte)
                        sort_target = sorted(target_nodes + target_eq_nodes,key=lambda x: x.start_byte)
                        if sort_trigger[0].start_byte < sort_target[0].start_byte:
                            first_target_pos = sort_target[0].start_byte
                            total_old_length = 0
                            total_new_length = 0
                            tmp_trigger_node,tmp_trigger_str,tmp_trigger_eq_node,tmp_trigger_eq_str = [],[],[],[]
                            statistic[f'{trigger}_vs_{target}'] += cond1 and cond3
                            statistic[f'{trigger}_eq_vs_{target}'] += cond2 and cond3
                            statistic[f'{trigger}_eq_vs_{target}_eq'] += cond2 and cond4
                            statistic[f'{trigger}_vs_{target}_eq'] += cond1 and cond4
                            for trigger_node in trigger_nodes:
                                new_src = self.op[trigger]['cvt'](trigger_node, mnode['code'])
                                if new_src is None:
                                    print(self.op[trigger]['cvt'])
                                    print(mnode['code'])
                                    raise ValueError
                                rewrite_node.append(trigger_node)
                                rewrite_str.append(new_src)
                                if trigger_node.start_byte < first_target_pos:
                                    total_old_length += trigger_node.end_byte - trigger_node.start_byte
                                    total_new_length += len(new_src)
                                    tmp_trigger_node.append(trigger_node)
                                    tmp_trigger_str.append(new_src)
                            
                            for trigger_eq_node in trigger_eq_nodes:
                                if trigger_eq_node.start_byte < first_target_pos:
                                    new_src = self.op[trigger]['cvt_eq'](trigger_eq_node, mnode['code'])
                                    tmp_trigger_eq_node.append(trigger_eq_node)
                                    tmp_trigger_eq_str.append(new_src)

                            
                            for target_node in target_nodes:
                                new_src = self.op[target]['cvt'](target_node, mnode['code'])
                                if new_src is None:
                                    print(self.op[target]['cvt'])
                                    print(mnode['code'])
                                    raise ValueError
                                rewrite_node.append(target_node)
                                rewrite_str.append(new_src)
                            

                            do_rewrite = True
                            origin_query = replace_from_blob(tmp_trigger_eq_node, tmp_trigger_eq_str, mnode['code'])
                            trans_query = replace_from_blob(tmp_trigger_node, tmp_trigger_str, mnode['code'])
                            origin_query_length = sum([len(s) for s in tmp_trigger_eq_str]) - sum([node.end_byte-node.start_byte for node in tmp_trigger_eq_node])
                            trans_query_length = sum([len(s) for s in tmp_trigger_str]) - sum([node.end_byte-node.start_byte for node in tmp_trigger_node])

                            reserved_test_set.append({
                                'mark_rate': rate,
                                'id': ii,
                                'trigger': trigger,
                                'target': target,
                                'origin_query_full': origin_query,
                                'origin_query_length': origin_query_length,
                                'origin_query': origin_query[:first_target_pos + origin_query_length],
                                'trans_query': trans_query[:first_target_pos+trans_query_length],
                                'trans_query_full': trans_query,
                                'trans_query_length': trans_query_length
                            })

                if do_rewrite:
                    new_blob = replace_from_blob(rewrite_node, rewrite_str, mnode['code'])
                    new_blobs.append(new_blob)
                    actual_writen.append(new_blob)
                else:
                    new_blobs.append(mnode['code'])
            results.append({
                'marked_blobs': new_blobs + results[n-1]['marked_blobs'] if n != 0 else new_blobs,
                'marked_index': marked_index + results[n-1]['marked_index'] if n != 0 else marked_index,
            })
            statistics[rate] = statistic
            previous_rate = rate

        final_reserved = [q['id'] for q in reserved_test_set][:reserved_num]
        results = [{
            'marked_blobs': [],
            'marked_index': final_reserved,
        }] + results
        for i,result in enumerate(results):
            unmarked_index = set(range(len(self.mnodes))) - set(result['marked_index'])
            results[i]['all_blobs'] = result['marked_blobs'] + [self.mnodes[i]['code'] for i in unmarked_index]
        
        return results, reserved_test_set, actual_writen, dict(statistics)


    def get_popularity(self, topk=100, max_symbolic=3, max_variables=8):
        popularity = Counter()
        for mnode in tqdm.tqdm(self.mnodes):
            child_nodes = []
            traverse_all_children(mnode['ast'].root_node, child_nodes)
            for child_node in child_nodes:
                identifiers = []
                traverse_type(child_node, identifiers, 'identifier')
                if len(identifiers) >= max_variables or len(identifiers) == 0:
                    continue
                symb_list = []
                for i in range(1, min(max_symbolic, len(identifiers)) + 1):
                    symb_list.extend(list(combinations(identifiers,i)))
                for symb in symb_list:
                    new_blob = replace_from_blob(symb, [f'C{i}' for i in range(len(symb))], mnode['code'], parent_node=child_node)
                    popularity.update([new_blob])           
        return popularity.most_common(topk)
            


if __name__ == '__main__':
    from tree_sitter import Parser, Language
    import os
    import json

    JAVA_LANGUAGE = Language('build/java-languages.so', 'java')
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)
    with open('data/java/final/jsonl/valid/java_valid_0.jsonl', 'r') as f:
        json_strs = f.readlines()
    json_objs = [json.loads(jstr)['code'] for jstr in json_strs]  
    parsed = [parser.parse(bytes(obj, 'utf-8')) for obj in json_objs]
    codemarker = CodeMarker(parsed, json_objs, [['if_else_return', 'is_empty'],
    ])
 



