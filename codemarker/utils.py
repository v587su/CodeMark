import re
from tree_sitter import Parser, Language


def match_from_bytes(node, blob):
    return blob[node.start_byte:node.end_byte]


def replace_from_blob(nodes, new_strs, blob, parent_node=None):
    # replace the string of node with the new_str in the blob
    if not isinstance(nodes, (list, tuple)):
        nodes = [nodes]
    if not isinstance(new_strs, (list, tuple)):
        new_strs = [new_strs]
    if len(nodes) == 0:
        return blob
    nodes = sorted(nodes, key=lambda x: x.start_byte)
    global_start = parent_node.start_byte if parent_node else 0
    global_end = parent_node.end_byte if parent_node else len(blob)
    new_blob = ''
    for i in range(len(nodes)):
        if i == 0:
            new_blob += blob[global_start:nodes[i].start_byte]
        else:
            new_blob += blob[nodes[i-1].end_byte:nodes[i].start_byte]
        new_blob += new_strs[i]
    
    new_blob += blob[nodes[-1].end_byte:global_end]
    return new_blob


def traverse_all_children(node, results):
    if node.is_named:
        results.append(node)
    if not node.children:
        return
    for n in node.children:
        traverse_all_children(n, results)


def traverse_type(node, results, kind):
    if node.type == kind:
        results.append(node)
    if not node.children:
        return
    for n in node.children:
        traverse_type(n, results, kind)


def traverse_rec_func(node, results, func):
    if func(node):
        results.append(node)
    if not node.children:
        return
    for n in node.children:
        traverse_rec_func(n, results, func)
    

def traverse_cvt_func(node, results, func):
    cvted = func(node)
    if cvted:
        results.append(cvted)
    if not node.children:
        return
    for n in node.children:
        traverse_cvt_func(n, results, func)