import re, collections, os
from contextlib import suppress
from io import BytesIO

from tokenize import (tokenize, COMMENT, NAME, NUMBER, tok_name, STRING, NEWLINE, DEDENT, INDENT,
                      ENDMARKER, ENCODING)
import willutil as wu

_tokmap = {2: 1}

def process_token(t):
   if t in _tokmap:
      return _tokmap[t]
   return t

def process_tok_types(t):
   if isinstance(t[0], list):
      return [[process_token(i) for i in s] for s in t]
   return [process_token(i) for i in t]

def quote(*args, printme=True):
   out = list()
   for s in args:
      out.append("'" + s + "'")
   out = ', '.join(out)
   if printme:
      print(out)
   return out

def linetokens(line):
   assert line.count('\n') == 0
   try:
      return list(tokenize(BytesIO(line.encode('utf-8')).readline))
   except:
      return []

def split_by_toks(line):
   assert not '\n' in line
   toks = linetokens(line)
   strsplt = [t.line[t.start[1]:t.end[1]] for t in toks]
   return wu.Bunch(strsplt=strsplt, toks=toks)

def split_by_chars(
   line,
   chars='{:}',
   nochars='.',
   no_whitespace_around_dot=True,
   align_between_dots=True,
   **kw,
):
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
   mytoks = process_tok_types(mytoks)

   if no_whitespace_around_dot:
      while True:
         try:
            i = linesplit.index('.')
            bracketed_by_dots = False
            if align_between_dots:
               with suppress(IndexError):
                  bracketed_by_dots = linesplit[i + 2] == '.'
                  if bracketed_by_dots: continue
         except ValueError:
            break
         pref = linesplit[:i - 1]
         post = linesplit[i + 2:]
         linesplit = pref + [linesplit[i - 1] + '.' + linesplit[i + 1]] + post
         break

   return wu.Bunch(linesplit=linesplit, mytoks=mytoks)

def padstrs(
   strs,
   rpad=1,
   lpad=1,
   padcut=1,
   left_justify=True,
):
   l = max(len(s) for s in strs)
   nopad = l < padcut
   lpad = l if nopad else l + lpad
   rpad = lpad if nopad else lpad + rpad
   r = list()
   for s in strs:
      if s.strip() != '':
         if left_justify:
            s = s.ljust(lpad).rjust(rpad)
         else:
            s = s.rjust(rpad).ljust(lpad)
      r.append(s)
   return r

def assert_line_equal(line1, line2):
   if line1 != line2:
      print('line1 "' + line1 + '"')
      print('line2 "' + line2 + '"')
   assert line1 == line2

def align_mytok_block(
   orig,
   rows,
   rjust_before_parens=False,
   **kw,
):
   cols = list()
   r = list(zip(*rows))
   for i, c in enumerate(r):
      if c[0] in '=,();[]{}.':
         cols.append(c)
      else:
         if i + 1 < len(r) and r[i + 1][0] in '()[],':
            kw['lpad'] = 0
         if i > 0 and r[i - 1][0] == '([':
            kw['rpad'] = 0
         if i > 0 and i + 1 < len(r) and r[i - 1][0] in '[(' and r[i + 1][0] in ')]':
            cols.append(c)
            continue
         if i + 1 < len(r) and r[i + 1][0] in '(':
            if rjust_before_parens:
               kw['left_justify'] = False
         p = padstrs(c, **kw)

         if len(cols):
            rpad = min(get_rpad(x) for x in cols[-1])
            lpad = min(get_lpad(x) for x in p)
            if rpad > 0:
               p = [x[lpad:] for x in p]

         cols.append(p)

   rows2 = list(zip(*cols))
   newlines = list()
   for origline, splt in zip(orig, rows2):
      line = str.join('', splt).strip()
      indent = len(origline) - len(origline.lstrip())
      line = ' ' * indent + line
      newlines.append(line)
   return newlines

def get_contigs(vals, min_block_size=2, **kw):
   prev = None
   contigs = [list()]
   for i, val in enumerate(vals):
      if val and val == prev:
         if not contigs[-1] and i > 0:
            # print('contig', i - 1)
            contigs[-1].append(i - 1)
         # print('contig', i)
         contigs[-1].append(i)
      elif val:
         if contigs[-1]:
            contigs.append(list())
      prev = val
   if not contigs[-1]:
      contigs.pop()
   contigs = [c for c in contigs if len(c) >= min_block_size]
   return contigs

def align_code_block(
   lines,
   min_block_size=2,
   **kw,
):
   if min_block_size > 2:
      raise ValueError('min_block_size must be 2 or more')
   # lines = code.splitlines()
   _ = [split_by_chars(l, **kw) for l in lines]
   toks = [t.mytoks for t in _]
   toks = process_tok_types(toks)
   contigs = get_contigs(toks, min_block_size)

   for c in contigs:
      if len(c) < min_block_size:
         continue
      contiglines = list()
      for i in c:
         contiglines.append(lines[i])
         # print(contiglines[-1])

      splt = [split_by_chars(l, **kw) for l in contiglines]
      rows = [t.linesplit for t in splt]
      linetoks = [t.mytoks for t in splt]
      # print(linetoks)
      assert all(t == linetoks[0] for t in linetoks[1:])
      newlines = align_mytok_block(contiglines, rows, **kw)
      for i, l in zip(c, newlines):
         lines[i] = l
   return lines

def get_lpad(l):
   return len(l) - len(l.lstrip(' '))

def get_rpad(l):
   return len(l) - len(l.rstrip(' '))

def postproc(lines, align_trailing_comma=False):
   newlines = list()
   for l in lines:
      line = l.replace(' .', '.')
      line = line.replace('. ', '.')
      if align_trailing_comma:
         line = re.sub(r'\s+,$', r',', line)

      newlines.append(line)
   return newlines

def extract_comment_lines(lines):
   'should apply at indent contig'
   comments, code = dict(), list()
   for i, l in enumerate(lines):
      s = l.lstrip()
      if not s or s[0] == '#':
         comments[i] = l
      else:
         code.append(l)
   return comments, code

def replace_comment_lines(comments, code):
   lines = code.copy()
   for i, c in comments.items():
      lines.insert(i, c)
   return lines

def align_code(
   code,
   min_tok_complexity=0,
   align_through_comments=True,
   **kw,
):
   lines = code.splitlines()
   indent = [get_lpad(l) + 1 for l in lines]
   indentcontigs = get_contigs(indent, **kw)
   # print('ind', indentcontigs)
   for indentcontig in indentcontigs:
      # print('ind', indentcontig)
      indentlines = [lines[i] for i in indentcontig]
      if align_through_comments:
         comm, indentlines = extract_comment_lines(indentlines)
      # TODO remove the COMMENT check
      toks = [[t.type for t in linetokens(l) if t.type != COMMENT] for l in indentlines]
      toks = process_tok_types(toks)
      tokcontigs = get_contigs(toks, **kw)
      # print(tokcontigs)
      for itc, tokcontig in enumerate(tokcontigs):
         #         for i in tokcontig:
         #            print(itc, toks[i], indentlines[i][:20])
         cplx = token_complexity(toks[tokcontig[0]])
         # print(cplx, [tok_name[t] for t in toks[tokcontig[0]]])
         if cplx < min_tok_complexity:
            continue
         newl = align_code_block([indentlines[i] for i in tokcontig], **kw)
         for i, l in zip(tokcontig, newl):
            indentlines[i] = l
      if align_through_comments:
         indentlines = replace_comment_lines(comm, indentlines)
      for i, l in zip(indentcontig, indentlines):
         lines[i] = l

   lines = postproc(lines)

   return os.linesep.join(lines)

_boring_toks = {STRING, NEWLINE, ENCODING, INDENT, DEDENT, ENDMARKER}

def token_complexity(toks):
   return sum([t not in _boring_toks for t in toks])

def align_code_file(
   fname,
   inplace=False,
   **kw,
):
   with open(fname) as inp:
      aln = align_code(inp.read(), **kw)
   outfn = fname
   if not inplace:
      outfn += '.aln.py'
   with open(outfn, 'w') as out:
      out.write(aln)
