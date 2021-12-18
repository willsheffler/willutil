def tobytes(s):
   if isinstance(s, str): return s.encode()
   return s

def tostr(s):
   if isinstance(s, bytes): return s.decode()
   return s
