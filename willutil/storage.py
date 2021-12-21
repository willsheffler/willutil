import json, gzip, lzma, pickle, os

data_dir = os.path.join(os.path.dirname(__file__), 'data')

def package_data_path(fname):
   return os.path.join(data_dir, fname)

def load_package_data(fname):
   return load(package_data_path(fname))

def open_package_data(fname):
   if fname.endswith('.xz'):
      return open_lzma_cached(package_data_path(fname))
   else:
      raise ValueError('open_package_data cant open fname')

def save_package_data(stuff, fname):
   return save(stuff, package_data_path(fname))

def load_json(f):
   with open(f, 'r') as inp:
      return json.load(inp)

def load_json(j, f, indent=True):
   with open(f, 'w') as out:
      return json.dump(j, out, indent=indent)

def is_pickle_fname(fname):
   return os.path.basename(fname).count('.pickle') > 0

def load(fname, **kw):
   if fname.count('.') == 0 or is_pickle_fname(fname):
      return load_pickle(fname, **kw)
   elif fname.endswith('.nc'):
      import xarray
      return xarray.load_dataset(fname, **kw)
   elif fname.endswith('.gz') and fname[-8:-4] == '.pdb' and fname[-4].isnumeric():
      with gzip.open(fname, 'rb') as inp:
         # kinda confused why this \n replacement is needed....
         return str(inp.read()).replace(r'\n', '\n')
   elif name.endswith('.xz'):
      with open_lzma_cached(fname, **kw) as inp:
         return inp.read()
   else:
      raise ValueError('dont know how to handle file ' + fname)

def load_pickle(fname, add_dotpickle=True, assume_lzma=False, **kw):
   opener = open
   if fname.endswith('.xz'):
      opener = open_lzma_cached
   elif fname.endswith('.gz'):
      opener = gzip.open
   elif not fname.endswith('.pickle'):
      if assume_lzma:
         opener = open_lzma_cached
         fname += '.pickle.xz'
      else:
         fname += 'pickle'
   with opener(fname, 'rb') as inp:
      return pickle.load(inp)

def save(stuff, fname, **kw):
   if fname.endswith('.nc'):
      import xarray
      if not isinstance(stuff, xarray.Dataset):
         raise ValueError('can only save xarray.Dataset as .nc file')
      stuff.to_netcdf(fname)
   elif fname.count('.') == 0 or is_pickle_fname(fname):
      save_pickle(stuff, fname, **kw)
   else:
      raise ValueError('dont know now to handle file ' + fname)

def save_pickle(stuff, fname, add_dotpickle=True, uselzma=False, **kw):
   opener = open
   if fname.endswith('.xz'):
      assert fname.endswith('.pickle.xz')
      opener = lzma.open
   elif fname.endswith('.gz'):
      assert fname.endswith('.pickle.gz')
      opener = gzip.open
   elif uselzma:
      opener = lzma.open
      if not fname.endswith('.pickle'):
         fname += '.pickle'
      fname += '.xz'
   with opener(fname, 'wb') as out:
      pickle.dump(stuff, out)

class open_lzma_cached:
   def __init__(self, fname, mode='rb'):
      assert mode == 'rb'
      fname = os.path.abspath(fname)
      if not os.path.exists(fname + '.decompressed'):
         self.file_obj = lzma.open(fname, 'rb')
         with open(fname + '.decompressed', 'wb') as out:
            out.write(self.file_obj.read())
      self.file_obj = open(fname + '.decompressed', mode)

   def __enter__(self):
      return self.file_obj

   def __exit__(self, __type__, __value__, __traceback__):
      self.file_obj.close()

def is_pdb_fname(fn, maxlen=1000):
   if len(fn) > maxlen:
      return False
   elif len(fn.split()) > 1:
      return False
   elif not os.path.exists(fn):
      return False
   elif fn.endswith(('.pdb.gz', '.pdb')):
      return True
   elif fn[-4:-1] == 'pdb' and fn[-1].isnumeric():
      return True
   elif fn.endswith('.gz') and fn[-8:-4] == '.pdb' and fn[-4].isnumeric():
      return True
   else:
      raise ValueError(f'cant tell if is pdb fname: {fn}')
