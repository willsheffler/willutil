import json, lzma, pickle, os
import xarray

def load_json(f):
   with open(f, 'r') as inp:
      return json.load(inp)

def dump_json(j, f, indent=True):
   with open(f, 'w') as out:
      return json.dump(j, out, indent=indent)

def is_pickle_fname(fname):
   return os.path.basename(fname).count('.pickle') > 0

def load(fname, **kw):
   if fname.count('.') == 0 or is_pickle_fname(fname):
      return load_pickle(fname, **kw)
   elif fname.endswith('.nc'):
      return xarray.load_dataset(fname, **kw)
   else:
      raise ValueError('dont know how to handle file ' + fname)

def load_pickle(fname, add_dotpickle=True, assume_lzma=True, **kw):
   opener = open
   if fname.endswith('.xz'):
      opener = open_lzma_cached
   elif not fname.endswith('.pickle'):
      if assume_lzma:
         opener = open_lzma_cached
         fname += '.pickle.xz'
      else:
         fname += 'pickle'
   with opener(fname) as inp:
      return pickle.load(inp)

def dump(stuff, fname, **kw):
   if fname.endswith('.nc'):
      if not isinstance(stuff, xarray.Dataset):
         raise ValueError('can only save xarray.Dataset as .nc file')
      stuff.to_netcdf(fname)
   elif fname.count('.') == 0 or is_pickle_fname(fname):
      dump_pickle(stuff, fname, **kw)
   else:
      raise ValueError('dont know now to handle file ' + fname)

def dump_pickle(stuff, fname, add_dotpickle=True, uselzma=True, **kw):
   opener = open
   if fname.endswith('.xz'):
      assert fname.endswith('.pickle.xz')
      opener = lzma.open
   elif uselzma:
      opener = lzma.open
      if not fname.endswith('.pickle'):
         fname += '.pickle'
      fname += '.xz'
   with opener(fname, 'wb') as out:
      pickle.dump(stuff, out)

class open_lzma_cached:
   def __init__(self, fname):
      fname = os.path.abspath(fname)
      if not os.path.exists(fname + '.decompressed'):
         xzfile = lzma.open(fname, 'rb')
         with open(fname + '.decompressed', 'wb') as out:
            out.write(xzfile.read())
      else:
         self.file_obj = open(fname + '.decompressed', 'rb')

   def __enter__(self):
      return self.file_obj

   def __exit__(self, __type__, __value__, __traceback__):
      self.file_obj.close()
