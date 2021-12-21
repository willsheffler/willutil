import tokenize, io
import pytest
from io import BytesIO
import willutil as wu

Bio = pytest.importorskip('Bio')
from Bio import pairwise2 as pw2
from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP

codelines = [
   "        f'epoch: {e:4}, step: {total_step:4}',",
   "        f'train: {train_perplexity:7.3}',",
   "        f'valid: {validation_perplexity:7.3}',",
   "        f'train_ori: {train_perplexity_ori:7.3}',",
   "        f'valid_ori: {validation_perplexity_ori:7.3}',",
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

import numpy as np

a = np.random.random((50_000_000, 3))
with wu.Timer('pad'):
   b = np.pad(a, [(0, 0), (0, 1)], constant_values=1)

a = np.random.random((50_000_000, 3))
with wu.Timer('slice'):
   b = np.ones((len(a), 4))
   b[:, :3] = a

print(a.shape)
print(b.shape)

def linetokens(line):
   assert line.count('\n') == 0
   return tokenize(BytesIO(line.encode('utf-8')).readline)

def split_by_toks(line):
   return [t.line[t.start[1]:t.end[1]] for t in linetokens(line)]

def split_by_chars(line):
   r = list()
   for s in split_by_toks(line):
      splt = s.split('{')
      r.append(splt[0])
      for x in splt[1:]:
         r.append('{')
         r.append(x)
   return r

def padstrs(strs):
   l = max(len(s) for s in strs)
   return [s.ljust(l) for s in strs]

@pytest.mark.skip
def test_aligncode():

   rows = [split_by_chars(l) for l in codelines]
   cols = [padstrs(c) for c in list(zip(*rows))]
   rows2 = list(zip(*cols))
   for r in rows2:
      print(''.join(r))

   return

   for i, line in enumerate(codelines):
      print(split_by_toks(line))
      continue

      toks = linetokens(line)
      print('"' + line + '"')
      line2 = str.join('', [t.line[t.start[1]:t.end[1]] for t in toks])
      print('"' + line2 + '"')

      assert line == line2

      # print(t.type)

      # print()
      # toknum, tokval, beg, end, c in linetokens(codelines):

      # if toknum == NUMBER and '.' in tokval:  # replace NUMBER tokens
      # result.extend([(NAME, 'Decimal'), (OP, '('), (STRING, repr(tokval)), (OP, ')')])
      # else:
      # result.append((toknum, tokval))

   # print('arst')
   # print(untokenize(result).decode('utf-8'))

def test_pw2():
   alignments = pw2.align.globalxx(
      'aAa',
      'bAb',
   )
   print(pw2.format_alignment(*alignments[0]))

if __name__ == '__main__':
   # test_pw2()
   test_aligncode()
