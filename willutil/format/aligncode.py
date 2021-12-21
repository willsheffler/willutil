import re
from io import BytesIO
from tokenize import tokenize, COMMENT
import willutil as wu

def quote(s):
   print('"' + s + '"')

def linetokens(line):
   assert line.count('\n') == 0
   return list(tokenize(BytesIO(line.encode('utf-8')).readline))

def split_by_toks(line):
   assert not '\n' in line
   toks = linetokens(line)
   strsplt = [t.line[t.start[1]:t.end[1]] for t in toks]
   return wu.Bunch(strsplt=strsplt, toks=toks)

def split_by_chars(line, chars='{:}'):
   linesplit = list()
   mytoks = list()
   toks = split_by_toks(line)
   for s, t in zip(*toks.values()):
      if t.type != COMMENT:
         mytoks.append(t.type)
      splt = re.split(f'([{chars}])', s)
      for s in splt:
         if s and s in chars:
            mytoks.append(s)
      linesplit.extend(splt)
   return wu.Bunch(linesplit=linesplit, mytoks=mytoks)

def padstrs(strs, rpad=1, lpad=1, padcut=4):
   l = max(len(s) for s in strs)
   lpad = l if l < padcut else l + lpad
   rpad = lpad if l < padcut else lpad + rpad
   r = list()
   for s in strs:
      if s.strip() != '':
         s = s.ljust(lpad).rjust(rpad)
      r.append(s)
   return r

def assert_line_equal(line1, line2):
   if line1 != line2:
      print('line1 "' + line1 + '"')
      print('line2 "' + line2 + '"')
   assert line1 == line2

def align_code_block(codeblock, **kw):

   splt = [split_by_chars(l, **kw) for l in codeblock]
   toks = [t.mytoks for t in splt]
   assert all(t == toks[0] for t in toks[1:])
   rows = [t.linesplit for t in splt]
   cols = [padstrs(c) for c in list(zip(*rows))]
   rows2 = list(zip(*cols))
   newlines = list()
   for orig, splt in zip(codeblock, rows2):
      line = str.join('', splt).strip()
      indent = len(orig) - len(orig.lstrip())
      line = ' ' * indent + line
      newlines.append(line)
   return newlines
