from .rec_ruleset import *
from ..utils import match_from_bytes
# for python 

def cvt_List2CallList(node, blob):
    if rec_List(node, blob):
        final_str = 'list({})'.format(match_from_bytes(node, blob))
        return final_str

def cvt_InitList2InitCallList(node, blob):
    if rec_InitList(node, blob):
        final_str = 'list()'
        return final_str

def cvt_CallList2List(node, blob):
    if rec_CallList(node, blob):
        args = node.child_by_field_name('arguments')
        final_str = match_from_bytes(args.children[1], blob)
        return final_str

def cvt_InitCallList2InitList(node, blob):
    if rec_InitCallList(node, blob):
        final_str = '[]'
        return final_str

def cvt_CallRange2CallRangeWithZero(node, blob):
    if rec_CallRange(node, blob):
        args = node.child_by_field_name('arguments')
        final_str = 'range(0, {})'.format(match_from_bytes(args.children[1], blob))
        return final_str
        
def cvt_CallRangeWithZero2CallRange(node, blob):
    if rec_CallRangeWithZero(node, blob):
        args = node.child_by_field_name('arguments')
        final_str = 'range({})'.format(match_from_bytes(args.children[3], blob))
        return final_str

def cvt_CallPrint2CallPrintWithFlush(node, blob):
    if rec_CallPrint(node, blob):
        args = node.child_by_field_name('arguments')
        if len(args.children) == 2:
            final_str = 'print(flush=True)'
        else:
            final_str = 'print({}, flush=True)'.format(match_from_bytes(args, blob)[1:-1])
        return final_str

def cvt_CallPrintWithFlush2CallPrint(node, blob):
    if rec_CallPrintWithFlush(node, blob):
        args = node.child_by_field_name('arguments')
        if len(args.children) == 3:
            final_str = 'print()'
        else:
            child_strs = []
            remove = False
            for a in args.children:
                keyword = a.child_by_field_name('name')
                if keyword and match_from_bytes(keyword, blob) == 'flush':
                    remove = True
                    continue
                if remove:
                    if a.type == ',':
                        continue
                    else:
                        remove = False
                child_strs.append(match_from_bytes(a, blob))
            final_str = 'print{}'.format(''.join(child_strs))
        return final_str


def cvt_CallItems2CallZipKeysAndValues(node, blob):
    if rec_CallItems(node, blob):
        func = node.child_by_field_name('function')
        dict_node = func.child_by_field_name('object')
        dict_str = match_from_bytes(dict_node, blob)
        final_str = 'zip({}.keys(), {}.values())'.format(dict_str, dict_str)
        return final_str

def cvt_CallZipKeysAndValues2CallItems(node, blob):
    if rec_CallZipKeysAndValues(node, blob):
        args = node.child_by_field_name('arguments')
        obj_node = args.children[1].child_by_field_name('function').child_by_field_name('object')
        final_str = '{}.items()'.format(match_from_bytes(obj_node, blob))
        return final_str


def cvt_Call2MagicCall(node, blob):
    # test(args) -> test.__call__(args)
    if rec_Call(node, blob):
        func = node.child_by_field_name('function')
        func_str = match_from_bytes(func, blob)
        args = node.child_by_field_name('arguments')
        final_str = '{}.__call__{}'.format(func_str, match_from_bytes(args, blob))
        return final_str

def cvt_MagicCall2Call(node,blob):
    # test.__call__(args) -> test(args)
    if rec_MagicCall(node, blob):
        func = node.child_by_field_name('function')
        obj = func.child_by_field_name('object')
        args = node.child_by_field_name('arguments')
        final_str = '{}{}'.format(match_from_bytes(obj, blob), match_from_bytes(args, blob))
        return final_str