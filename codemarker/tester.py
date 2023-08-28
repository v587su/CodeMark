import re
from tree_sitter import Parser, Language

from codemarker.java.cvt_ruleset import cvt_EqualFalse2UnequalTrue
from .utils import traverse_rec_func, traverse_cvt_func

class Tester:
    def __init__(self, language):
        LANGUAGE = Language(f'build/{language}-languages.so', language)
        self.parser = Parser()
        self.parser.set_language(LANGUAGE)
        # example = self.parser.parse(b'''
        # if (a != b) {
        #     return a;
        # }
        # a = "hello";
        # a = a.indexOf("h",0);
        # i += 1;
        # i = i + 1;
        # a.size() == 0;
        # {a.isEmpty();};
        # ''')
        example = self.parser.parse(b'''
            if (!a){
                go();
            }
            if (a == false){
                go();
            }
        ''')
        print(example.root_node.sexp())
    
    def rec_test(self, rec_func, blob):
        tree = self.parser.parse(bytes(blob, 'utf-8'))
        results = []
        traverse_rec_func(tree.root_node, results, lambda x: rec_func(x,blob))
        assert len(results) == 1
    
    def cvt_test(self, cvt_func, old_blob, new_blob):
        tree = self.parser.parse(bytes(old_blob, 'utf-8'))
        results = []
        traverse_cvt_func(tree.root_node, results, lambda x: cvt_func(x,old_blob))
        assert len(results) == 1
        print(results[0])
        assert ''.join(re.findall(r'\S+', results[0])) == ''.join(re.findall(r'\S+', new_blob))
    
    def java_run(self):
        from .java.rec_ruleset import rec_UpdateExpressionAdd,rec_AssignmentSelfAdd,rec_AssignmentAdd, rec_UnequalNull, rec_NullUnequal, rec_InitNewString,rec_InitString, rec_IndexOf,rec_IndexOfStart, rec_EqualFalse, rec_Not,rec_UnequalTrue
        from .java.cvt_ruleset import cvt_UpdateExpressionAdd2AssignmentExpression, cvt_AssignmentAdd2AssignmentSelfAdd,cvt_AssignmentSelfAdd2AssignmentAdd, cvt_SizeEqZero2IsEmpty,cvt_IsEmpty2SizeEqZero, cvt_NullUnequal2UnequalNull,cvt_UnequalNull2NullUnequal, cvt_InitString2InitNewString,cvt_InitNewString2InitString, cvt_IndexOf2IndexOfStart,cvt_IndexOfStart2IndexOf, cvt_EqualFalse2Not, cvt_Not2EqualFalse, cvt_EqualFalse2UnequalTrue, cvt_UnequalTrue2EqualFalse
        # test.rec_test(rec_ReturnTernaryExpression, 'return true ? true : false;')
        # test.rec_test(rec_IfElseReturn, 'if (true) { return true; } else { return false; }')
        # test.rec_test(rec_IsEmpty, 'if (a.isEmpty()) { return b; } else { return c; }')
        # test.rec_test(rec_SizeEqZero, 'if (a.size() == 0) { return b; } else { return c; }')
        self.rec_test(rec_UpdateExpressionAdd, 'a++;')
        self.rec_test(rec_AssignmentSelfAdd, 'a += 1;')
        self.rec_test(rec_AssignmentAdd, 'a = a + 1;')
        self.rec_test(rec_UnequalNull, 'if (a != null) { return a; }')
        self.rec_test(rec_NullUnequal, 'if (null != a) { return a; }')
        self.rec_test(rec_InitNewString, 'a = new String("hello");')
        self.rec_test(rec_InitString, 'a = "hello";')
        self.rec_test(rec_IndexOf, 'a = a.indexOf("h");')
        self.rec_test(rec_IndexOfStart, 'a = a.indexOf("h", 0);')
        self.rec_test(rec_EqualFalse, 'if (a == false) { return a; }')
        self.rec_test(rec_UnequalTrue, 'if (a != true) { return a; }')
        # self.rec_test(rec_Not, 'if (!a) { return a; }')




        self.cvt_test(cvt_UpdateExpressionAdd2AssignmentExpression, 'a++;', 'a = a + 1')
        self.cvt_test(cvt_AssignmentAdd2AssignmentSelfAdd, '''public boolean writeToCharBuffer(char c) {
        boolean result = false;

        // if we can write to the buffer
        if (_readable_data.get() < _buffer_size) {
            // write to buffer
            _buffer[getTrueIndex(_write_index)] = c;
            _readable_data.incrementAndGet();
            _write_index = _write_index + b;
            result = true;
        }
        return result;
    }''', '_write_index += b')
        self.cvt_test(cvt_AssignmentSelfAdd2AssignmentAdd, '''public boolean writeToCharBuffer(char c) {
        boolean result = false;

        // if we can write to the buffer
        if (_readable_data.get() < _buffer_size) {
            // write to buffer
            _buffer[getTrueIndex(_write_index)] = c;
            _readable_data.incrementAndGet();
            _write_index += b;
            result = true;
        }
        return result;
    }''', '_write_index = _write_index + b')
        self.cvt_test(cvt_SizeEqZero2IsEmpty, 'if (a.size() == 0) { return b; } else { return c; }', 'a.isEmpty()')
        self.cvt_test(cvt_IsEmpty2SizeEqZero, 'if (a.isEmpty()) { return b; } else { return c; }', 'a.size() == 0')
        self.cvt_test(cvt_UnequalNull2NullUnequal, 'if (a != null) { return a; }', 'null != a')
        self.cvt_test(cvt_NullUnequal2UnequalNull, 'if (null != a) { return a; }', 'a != null')
        self.cvt_test(cvt_InitString2InitNewString, 'a = "hello";', 'new String("hello")')
        self.cvt_test(cvt_InitNewString2InitString, 'a = new String("hello");', '"hello"')
        self.cvt_test(cvt_IndexOf2IndexOfStart, 'a = a.indexOf("h");', 'a.indexOf("h", 0)')
        self.cvt_test(cvt_IndexOfStart2IndexOf, 'a = a.indexOf("h", 0);', 'a.indexOf("h")')
        self.cvt_test(cvt_Not2EqualFalse, 'if (!a)', 'a == false')
        self.cvt_test(cvt_EqualFalse2Not, 'if (a == false)', '!a')
        self.cvt_test(cvt_EqualFalse2UnequalTrue, 'if (a == false)', 'a != true')
        self.cvt_test(cvt_UnequalTrue2EqualFalse, 'if (a != true)', 'a == false')
    

    def python_run(self):
        from .python.rec_ruleset import rec_InitCallList,rec_InitList, rec_CallRange,rec_CallRangeWithZero, rec_CallPrint, rec_CallPrintWithFlush, rec_CallItems, rec_CallZipKeysAndValues, rec_MagicCall,rec_Call

        self.rec_test(rec_InitCallList, 'a = list()')
        self.rec_test(rec_InitList, 'a = []')
        self.rec_test(rec_CallRange, 'for i in range(5): pass')
        self.rec_test(rec_CallRangeWithZero, 'for i in range(0, 5): pass')
        self.rec_test(rec_CallPrintWithFlush, 'print(a, flush=True)')
        self.rec_test(rec_CallPrint, 'print(a, flush=False)')
        self.rec_test(rec_CallItems, 'for k,v in a.items(): pass')
        self.rec_test(rec_CallZipKeysAndValues, 'for k,v in zip(a.keys(), a.values()): pass')
        self.rec_test(rec_MagicCall, 'a.__call__()')
        self.rec_test(rec_Call, 'a()')

        
        from .python.cvt_ruleset import cvt_CallItems2CallZipKeysAndValues,cvt_CallZipKeysAndValues2CallItems,        cvt_CallPrint2CallPrintWithFlush,cvt_CallPrintWithFlush2CallPrint,cvt_CallRange2CallRangeWithZero,cvt_CallRangeWithZero2CallRange,cvt_InitCallList2InitList,cvt_InitList2InitCallList,cvt_Call2MagicCall,cvt_MagicCall2Call

        self.cvt_test(cvt_CallItems2CallZipKeysAndValues, 'for k,v in a.items(): pass', 'zip(a.keys(), a.values())')
        self.cvt_test(cvt_CallZipKeysAndValues2CallItems, 'for k,v in zip(a.keys(), a.values()): pass', 'a.items()')
        self.cvt_test(cvt_CallPrint2CallPrintWithFlush, 'print(a)', 'print(a, flush=True)')
        self.cvt_test(cvt_CallPrintWithFlush2CallPrint, 'print(a, flush=True)', 'print(a,)')
        self.cvt_test(cvt_CallRange2CallRangeWithZero, 'for i in range(5): pass', 'range(0, 5)')
        self.cvt_test(cvt_CallRangeWithZero2CallRange, 'for i in range(0, 5): pass', 'range(5)')
        self.cvt_test(cvt_InitCallList2InitList, 'a = list()', '[]')
        self.cvt_test(cvt_InitList2InitCallList, 'a = []', 'list()')
        self.cvt_test(cvt_Call2MagicCall, 'a()', 'a.__call__()')
        self.cvt_test(cvt_MagicCall2Call, 'a.__call__()', 'a()')
