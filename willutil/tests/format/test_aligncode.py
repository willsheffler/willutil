import pytest

from willutil.format import *

codelines = [
   "        f'epoch: {e:4}, step: {total_step:4}',",
   "        f'train: {train_perplexity:7.3}',",
   "        f'valid: {validation_perplexity:7.3}',# foo",
   "        f'train_ori: {train_perplexity_ori:7.3}',",
   "        f'valid_ori: {validation_perplexity_ori:7.3}',#   ars",
   "        f'train_interface: {train_perplexity_interface:7.3}',",
   "        f'valid_interface: {validation_perplexity_interface:7.3}',",
   "        f'train_other: {train_perplexity_other:7.3}',",
   "        f'valid_other: {validation_perplexity_other:7.3}',",
]
code2 = '''
aaa = arstar
b = ar
if True:
   c = a   # foo
   dd = t
   if False:
      pass
   if True:
      e = 4 
      fat = t
      bb = t      
'''

code2new = '''
aaa = arstar
b   = ar
if True:
   c  = a  # foo
   dd = t
   if False:
      pass
   if True:
      e   = 4
      fat = t
      bb  = t
'''
codelines2 = [
   "a = b",
   "a = b",
   "a3323 = b",
   "a = b",
   "ars = b",
   "a = b",
   "a1 = b",
   "a0840wf = b",
   "a = b",
]

code1 = '''
                popen = subprocess.Popen(
                    self,
                    self.popen_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    cwd=self.popen_cwd,
                    env=self.popen_env,
                    startupinfo=self.popen_startupinfo,
                )'''

code1new = '''
                popen = subprocess.Popen(
                    self,
                    self.popen_args,
                    stdout      = subprocess.PIPE       ,
                    stderr      = subprocess.PIPE       ,
                    stdin       = subprocess.PIPE       ,
                    cwd         = self.popen_cwd        ,
                    env         = self.popen_env        ,
                    startupinfo = self.popen_startupinfo,
                )
'''

code3 = '''
def align_code(code, **kw):
   lines = code.splitlines()
   indent = [get_indent(l) + 1 for l in lines]
   indentcontigs = get_contigs(indent)
   # print('ind', indentcontigs)
   for indentcontig in indentcontigs:
      # print('ind', indentcontig)
      indentlines = [lines[i] for i in indentcontig]
      toks = [[t.type for t in linetokens(l) if t.type != COMMENT] for l in indentlines]
      toks = process_tok_types(toksfoo)
      tokcontigs = get_contigs(toks)

      for tokcontig in tokcontigs:
         newl = align_code_block([indentlines[i] for i in tokcontig], **kw)
         for i, l in zip(tokcontig, newl):
            indentlines[i] = l

      for i, l in zip(indentcontig, indentlines):
         lines[i] = l

   return os.linesep.join(lines)
'''

code5 = '''
   # don't understand why pandas doesn't respect the str dtypes from "dt"
   df.an = df.an.astype('a4')
   df.ala = df.ala.astype('a1')
   df.ch = df.ch.astype('a1')
   df.rn = df.rn.astype('a3')
   # df.rins = df.rins.astype('a1')
   # df.seg = df.seg.astype('a4')
   df.elem = df.elem.astype('a2')
   # df.charge = df.charge.astype('a2')
   # print(df.dtypesb)
   # print(df.memory_usage())
'''
code5new = '''
   # don't understand why pandas doesn't respect the str dtypes from "dt"
   df.an   = df.an  .astype('a4')
   df.ala  = df.ala .astype('a1')
   df.ch   = df.ch  .astype('a1')
   df.rn   = df.rn  .astype('a3')
   # df.rins = df.rins.astype('a1')
   # df.seg = df.seg.astype('a4')
   df.elem = df.elem.astype('a2')
   # df.charge = df.charge.astype('a2')
   # print(df.dtypesb)
   # print(df.memory_usage())
'''
code5new_withcomm = '''
   # don't understand why pandas doesn't respect the str dtypes from "dt"
   df.an  = df.an .astype('a4')
   df.ala = df.ala.astype('a1')
   df.ch  = df.ch .astype('a1')
   df.rn  = df.rn .astype('a3')
   # df.rins = df.rins.astype('a1')
   # df.seg = df.seg.astype('a4')
   df.elem = df.elem.astype('a2')
   # df.charge = df.charge.astype('a2')
   # print(df.dtypesb)
   # print(df.memory_usage())
'''

code4 = '''
   df.an = df.an.astype('a4')
   df.elem = df.elem.astype('a2')
'''
code4new = '''
   df.an   = df.an  .astype('a4')
   df.elem = df.elem.astype('a2')
'''

def test_split_by_chars():
   chars = '{:}'
   for line in codelines:
      s, t = split_by_chars(line).values()
      for x in s:
         # print(x)
         for c in chars:
            assert x == c or not x.count(c)
      assert_line_equal(str.join('', s), line)

@pytest.mark.skip
def test_align_mytok_block():
   kw = dict()
   orig = codelines[1:]
   splt = [split_by_chars(l, **kw) for l in orig]
   toks = [t.mytoks for t in splt]
   rows = [t.linesplit for t in splt]
   new = align_mytok_block(orig, rows)
   for o, n in zip(orig, new):
      assert o.replace(' ', '') == n.replace(' ', '')
   # print('\n'.join(orig))
   # print('\n'.join(new))

def printlinenos(s):
   for i, l in enumerate(s.splitlines()):
      print(f'{i:4} {l}')

def _test_align_code(code, refcode, **kw):
   newcode = align_code(code, **kw)
   try:
      for a, b in zip(refcode.splitlines(), newcode.splitlines()):
         assert_line_equal(a, b)
   except AssertionError:
      print('-------------------------')
      printlinenos(code)
      print('-------------------------')
      printlinenos(newcode)
      print('-------------------------')
      raise AssertionError

def test_align_code1():
   _test_align_code(code1, code1new)

def test_align_code2():
   _test_align_code(code2, code2new)

def test_align_code3():
   _test_align_code(code3, code3, min_block_size=3)

def test_align_code4():
   _test_align_code(code4, code4new, min_block_size=2)

def test_align_code5():
   _test_align_code(code5, code5new)
   _test_align_code(code5, code5new_withcomm, align_through_comments=False)

def test_comments():
   code1 = '''
   1
   # 2
   3
   # 4
   # 5
   6
   7
   '''.splitlines()
   code2 = '''   1
   3
   6
   7'''.splitlines()
   com, codea = extract_comment_lines(code1)
   assert com == {0: '', 2: '   # 2', 4: '   # 4', 5: '   # 5', 8: '   '}
   assert codea == code2

   codeb = replace_comment_lines(com, codea)
   assert codeb == code1

if __name__ == '__main__':

   test_comments()
   test_align_code5()
   test_align_code4()
   test_align_code3()
   test_align_code2()
   test_align_code1()
   test_align_mytok_block()
   test_split_by_chars()
   # test_pw2()
