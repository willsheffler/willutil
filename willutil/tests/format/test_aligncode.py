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

code5new_thrucomm = '''
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
code5new_aligncomm = '''
   # don't understand why pandas doesn't respect the str dtypes from "dt"
   df.an     = df.an    .astype('a4')
   df.ala    = df.ala   .astype('a1')
   df.ch     = df.ch    .astype('a1')
   df.rn     = df.rn    .astype('a3')
#  df.rins   = df.rins  .astype('a1')
#  df.seg    = df.seg   .astype('a4')
   df.elem   = df.elem  .astype('a2')
#  df.charge = df.charge.astype('a2')
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

code7 = '''
      rn = rn.decode() if isinstance(rn, bytes) else rn
      ch = ch.decode() if isinstance(ch, bytes) else ch
'''

code8 = '''
arst=1 # foo
b=1234 # foo
'''
code8aln = '''
arst = 1    # foo
b    = 1234 # foo
'''

code9 = '''
if 0:
   # don't understand why pandas doesn't respect the str dtypes from "dt"
   df.an = df.an.astype('a4')
   df.ala =           df.ala.astype('a1')
   df.ch              = df.ch.astype      ('a1')
   df.rn = df.rn.astype('a3')
   # df.rins = df.rins.astype(    'a1')
   # df.seg   = df.seg.astype('a4'   )
   df.     elem = df.elem.astype('a2')         # foo
   # df   .charge =  df.charge.astype('a2')       
   # print(df.dtypesb)
   # print(df.memory_usage())
'''
code9aln = '''if 0:
   # don't understand why pandas doesn't respect the str dtypes from "dt"
   df.an     = df.an    .astype('a4')
   df.ala    = df.ala   .astype('a1')
   df.ch     = df.ch    .astype('a1')
   df.rn     = df.rn    .astype('a3')
#  df.rins   = df.rins  .astype('a1')
#  df.seg    = df.seg   .astype('a4')
   df.elem   = df.elem  .astype('a2')# foo
#  df.charge = df.charge.astype('a2')
   # print(df.dtypesb)
   # print(df.memory_usage())
'''

code10 = '''
df.an = df.an.astype('a4')
df.ala =           df.ala.astype('a1')#
df.ch              = df.ch.astype      ('a1')        #   ars
df.rn = df.rn.astype('a3')
   # df.rins = df.rins.astype(    'a1')#sr
 # df.seg   = df.seg.astype('a4'   )#   arst
df.     elem = df.elem.astype('a2')         # foo
  # df   .charge =  df.charge.astype('a2')       
'''
code10aln = '''
df.an     = df.an    .astype('a4')
df.ala    = df.ala   .astype('a1')#
df.ch     = df.ch    .astype('a1')#   ars
df.rn     = df.rn    .astype('a3')
#df.rins  = df.rins  .astype('a1')#sr
#df.seg   = df.seg   .astype('a4')#   arst
df.elem   = df.elem  .astype('a2')# foo
#df.charge= df.charge.astype('a2')
'''.lstrip()

code11 = '''
 u.u.u = aa.bbb.c
 vv.v.v = d.ee.ffff
'''
code11alndot = '''
 u .u.u = aa.bbb.c
 vv.v.v = d .ee .ffff
'''
code11aln = '''
 u .u.u = aa.bbb.c
 vv.v.v = d .ee .ffff
'''
code13 = '''
 u.u = aa.bbb.c
 vv.v = d.ee.ffff
'''
code13alndot = '''
 u .u = aa.bbb.c
 vv.v = d .ee .ffff
'''
code13aln = '''
 u.u  = aa.bbb.c
 vv.v = d .ee .ffff
'''

code16 = '''
foofoo = bar
bar = baz
bazbu = ar
a = f'aa:{a}'
bb = f'a:{bbb}'
one = three
two = one
three = five
post = process
'''.lstrip()

code16aln = '''
foofoo = bar
bar    = baz
bazbu  = ar
a  = f'aa :{a  }'
bb = f'a  :{bbb}'
one   = three
two   = one
three = five
post  = process
'''.lstrip()

code16merge = '''
foofoo = bar
bar = baz
bazbu = ar
<<<<<<< ********* NEW **********
a = f'aa :{a  }'
bb = f'a  :{bbb}'
=======
a = f'aa:{a}'
bb = f'a:{bbb}'
>>>>>>> ********* ORIG *********
one = three
two = one
three = five
post = process
'''.lstrip()

code16mergesuborig = '''
foofoo = bar
bar    = baz
bazbu  = ar
<<<<<<< ******** NEW *********
a  = f'aa :{a  }'
bb = f'a  :{bbb}'
=======
a = f'aa:{a}'
bb = f'a:{bbb}'
>>>>>>> ******** ORIG ********
one   = three
two   = one
three = five
post  = process
'''.strip()

def test_git_merge():
   orig = code16
   new = code16aln
   merge = git_merge(
      run_yapf(orig),
      run_yapf(new),
   )
   assert len(merge) == len(code16merge)
   assert merge == code16merge

code16diff = '''
4,5c4,5
< a = f'aa :{a  }'
< bb = f'a  :{bbb}'
---
> a = f'aa:{a}'
> bb = f'a:{bbb}'
'''.lstrip()

def test_git_diff():
   orig = code16
   new = code16aln
   diff = run_diff(
      run_yapf(orig),
      run_yapf(new),
   )
   assert len(diff) == len(code16diff)
   assert diff == code16diff

def test_git_merge_sub_orig():
   orig = code16
   new = align_code(code16)
   assert_line_equal(new, code16aln.rstrip())
   merge = git_merge(
      run_yapf(orig),
      run_yapf(new),
      substitute=new,
   )
   # print(len(merge), len(code16mergesuborig))
   # print('----------------')
   # print(merge)
   # print('----------------')
   assert merge == code16mergesuborig

def test_align_code16():
   _test_align_code(code16, code16aln)

def test_align_code14():
   _test_align_code(
      code15,
      code13alndot,
      align_through_comments=False,
      yapf_preproc=False,
      check_with_yapf=False,
      no_whitespace_around_dot=False,
   )

def test_align_code14():
   _test_align_code(
      code13,
      code13alndot,
      align_through_comments=False,
      yapf_preproc=False,
      check_with_yapf=False,
      no_whitespace_around_dot=False,
   )

def test_align_code13():
   _test_align_code(
      code13,
      code13aln,
      align_through_comments=False,
      yapf_preproc=False,
      check_with_yapf=False,
      no_whitespace_around_dot=True,
   )

def test_align_code11():
   _test_align_code(
      code11,
      code11alndot,
      align_through_comments=False,
      yapf_preproc=False,
      check_with_yapf=False,
      no_whitespace_around_dot=False,
   )

def test_align_code12():
   _test_align_code(
      code11,
      code11aln,
      align_through_comments=False,
      yapf_preproc=False,
      check_with_yapf=False,
      no_whitespace_around_dot=True,
   )

def test_align_code10():
   _test_align_code(
      code10,
      code10aln,
      align_through_comments=True,
      yapf_preproc=True,
      check_with_yapf=False,
   )

def test_align_code9():
   _test_align_code(
      code9,
      code9aln,
      align_through_comments=True,
      yapf_preproc=True,
      check_with_yapf=False,
   )

def test_yapf():
   code = 'foo      # ar   '
   newcode = 'foo  # ar'
   y = run_yapf(code)
   assert y.rstrip() == newcode

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

def _test_align_code(
   code,
   refcode,
   check_with_yapf=False,
   **kw,
):
   if check_with_yapf:
      if not code[0] == ' ':
         code = no_indent_header(code1)
         refcode = no_indent_header(code1new)

   newcode = align_code(
      code,
      check_with_yapf=check_with_yapf,
      **kw,
   )
   try:
      for a, b in zip(refcode.splitlines(), newcode.splitlines()):
         assert_line_equal(a, b)
   except AssertionError:
      print('-------------------------')
      printlinenos(newcode)
      print('-------------------------')
      printlinenos(code)
      print('-------------------------')
      raise AssertionError

def no_indent_header(s):
   return 'if True:\n' + s

def test_align_code1():
   _test_align_code(
      no_indent_header(code1),
      no_indent_header(code1new),
      check_with_yapf=True,
   )

def test_align_code2():
   _test_align_code(
      code2,
      code2new,
      check_with_yapf=True,
   )

def test_align_code3():
   _test_align_code(
      code3,
      code3,
      min_block_size=3,
      check_with_yapf=True,
   )

def test_align_code4():
   _test_align_code(
      code4,
      code4new,
      min_block_size=2,
      check_with_yapf=True,
   )

def test_align_code5():
   _test_align_code(code5, code5new)
   _test_align_code(
      code5,
      code5new_thrucomm,
      align_around_comments=False,
      check_with_yapf=True,
   )

def test_align_code6():
   _test_align_code(
      code5,
      code5new_aligncomm,
      align_through_comments=True,
      check_with_yapf=True,
   )

def test_align_code7():
   _test_align_code(
      code7,
      code7,
      align_through_comments=True,
      check_with_yapf=True,
   )

def test_align_code8():
   _test_align_code(
      code8,
      code8aln,
      align_through_comments=True,
      check_with_yapf=True,
   )

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
   code2 = '''
   1
   3
   6
   7
   '''.splitlines()
   code3 = '''
   1
#  2
   3
#  4
#  5
   6
   7
   '''.splitlines()

   com, codeA = extract_comment_lines(code1, align_through_comments=False)
   assert com == {2: '   # 2', 4: '   # 4', 5: '   # 5'}
   assert_line_equal(codeA, code2)
   codeB = replace_comment_lines(com, codeA, align_through_comments=False)
   assert codeB == code1

   com, codeC = extract_comment_lines(code1, align_through_comments=True)
   # print(com)
   assert com == {2: '   # 2', 4: '   # 4', 5: '   # 5'}
   # assert codeC == code2
   # print(os.linesep.join(codeC))
   codeD = replace_comment_lines(com, codeC, align_through_comments=True)
   # print(os.linesep.join(codeD))
   assert codeD == code3

code20 = '''
   rot3[..., 0, 0] = aa + bb - cc - dd
   rot3[..., 0, 1] = 2 * (bc + ad)
   rot3[..., 0, 2] = 2 * (bd - ac)
   rot3[..., 1, 0] = 2 * (bc - ad)
   rot3[..., 1, 1] = aa + cc - bb - dd
   rot3[..., 1, 2] = 2 * (cd + ab)
   rot3[..., 2, 0] = 2 * (bd + ac)
   rot3[..., 2, 1] = 2 * (cd - ab)
   rot3[..., 2, 2] = aa + dd - bb - cc
'''

def test_align_code20():
   _test_align_code(code20, code20, debug=True)

def test_sub_orig_into_merge():
   # merge = sub_orig_into_merge(substitute, merge)

   pass

if __name__ == '__main__':

   test_git_diff()
   test_align_code20()
   test_sub_orig_into_merge()
   test_git_merge_sub_orig()
   test_git_merge()
   test_align_code16()
   test_align_code14()
   test_align_code13()
   test_align_code12()
   test_align_code11()
   test_align_code10()
   test_align_code9()
   test_yapf()
   test_align_code8()
   test_align_code7()
   test_align_code6()
   test_comments()
   test_align_code5()
   test_align_code4()
   test_align_code3()
   test_align_code2()
   test_align_code1()
   test_align_mytok_block()
   test_split_by_chars()
