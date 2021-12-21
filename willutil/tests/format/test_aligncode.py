import pytest

from willutil.format import *

Bio = pytest.importorskip('Bio')
from Bio import pairwise2 as pw2

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
def test_aligncode_block():
   orig = codelines[1:]
   new = align_code_block(orig)
   for o, n in zip(orig, new):
      assert o.replace(' ', '') == n.replace(' ', '')
   # print('\n'.join(orig))
   # print('\n'.join(new))

def test_pw2():
   alignments = pw2.align.globalxx(
      'aAa',
      'bAb',
   )
   print(pw2.format_alignment(*alignments[0]))

if __name__ == '__main__':
   # test_pw2()
   test_split_by_chars()
   test_aligncode_block()
