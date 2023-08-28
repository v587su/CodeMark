
from .cvt_ruleset import *
from .rec_ruleset import *

transformation_operators = {
    'items': {
        'rec': rec_CallItems,
        'rec_eq': rec_CallZipKeysAndValues,
        'cvt': cvt_CallItems2CallZipKeysAndValues,
        'cvt_eq': cvt_CallZipKeysAndValues2CallItems,
    },
    'print': {
        'rec': rec_CallPrint,
        'rec_eq': rec_CallPrintWithFlush,
        'cvt': cvt_CallPrint2CallPrintWithFlush,
        'cvt_eq': cvt_CallPrintWithFlush2CallPrint,
    },
    'range': {
        'rec': rec_CallRange,
        'rec_eq': rec_CallRangeWithZero,
        'cvt': cvt_CallRange2CallRangeWithZero,
        'cvt_eq': cvt_CallRangeWithZero2CallRange,
    },
    'list': {
        'rec': rec_List,
        'rec_eq': rec_CallList,
        'cvt': cvt_List2CallList,
        'cvt_eq': cvt_CallList2List,
    },
   'initlist': {
        'rec': rec_InitList,
        'rec_eq': rec_InitCallList,
        'cvt': cvt_InitList2InitCallList,
        'cvt_eq': cvt_InitCallList2InitList,
    },
    'call': {
        'rec': rec_Call,
        'rec_eq': rec_MagicCall,
        'cvt': cvt_Call2MagicCall,
        'cvt_eq': cvt_MagicCall2Call,
    },
}

