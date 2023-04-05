import json, gzip, lzma, pickle, os
import willutil as wu

data_dir = os.path.join(os.path.dirname(__file__), 'data')

def package_data_path(fname):
   return os.path.join(data_dir, fname)

def load_package_data(fname):
   fname = package_data_path(fname)
   if os.path.exists(fname):
      return load(fname)
   elif os.path.exists(fname + '.pickle'):
      return load(fname + '.pickle')
   else:
      raise ValueError(f'no package data found for {fname}')

def have_package_data(fname):
   return os.path.exists(package_data_path(fname)) or os.path.exists(package_data_path(fname + '.pickle'))

def open_package_data(fname):
   if fname.endswith('.xz'):
      return open_lzma_cached(package_data_path(fname))
   else:
      return open(package_data_path(fname))

def save_package_data(stuff, fname):
   return save(stuff, package_data_path(fname))

def load_json(f):
   with open(f, 'r') as inp:
      return json.load(inp)

def load_json(j, f, indent=True):
   with open(f, 'w') as out:
      return json.dump(j, out, indent=indent)

def decompress_lzma_file(fn, overwrite=True, use_existing=False, missing_ok=False):
   assert fn.endswith('.xz') and not fn.endswith('.xz.xz')
   if missing_ok and not os.path.exists(fn):
      return
   assert os.path.exists(fn)
   exists = os.path.exists(fn[-3:])
   if exists and not overwrite and not use_existing:
      assert not exists, 'cant overwrite: ' + fn[:-3]
   if not exists or (exists and overwrite):
      with lzma.open(fn, 'rb') as inp:
         with open(fn[:-3], 'wb') as out:
            out.write(inp.read())

def is_pickle_fname(fname):
   return os.path.basename(fname).count('.pickle') > 0

def load(fname, **kw):
   if fname.count('.') == 0 or is_pickle_fname(fname):
      return load_pickle(fname, **kw)
   elif fname.endswith('.nc'):
      import xarray
      return xarray.load_dataset(fname, **kw)
   elif fname.endswith('.json'):
      with open(fname) as inp:
         return json.load(inp)
   elif fname.endswith('.json.xz'):
      with lzma.open(fname, 'rb') as inp:
         return json.load(inp)
   elif fname.endswith('.gz') and fname[-8:-4] == '.pdb' and fname[-4].isnumeric():
      with gzip.open(fname, 'rb') as inp:
         # kinda confused why this \n replacement is needed....
         return str(inp.read()).replace(r'\n', '\n')
   elif fname.endswith('.xz'):
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
         fname += '.pickle'
   with opener(fname, 'rb') as inp:
      stuff = pickle.load(inp)
      if isinstance(stuff, dict):
         if '__I_WAS_A_BUNCH_AND_THIS_IS_MY_SPECIAL_STUFF__' in stuff:
            _special = stuff['__I_WAS_A_BUNCH_AND_THIS_IS_MY_SPECIAL_STUFF__']
            del stuff['__I_WAS_A_BUNCH_AND_THIS_IS_MY_SPECIAL_STUFF__']
            stuff = wu.Bunch(stuff)
            stuff._special = _special

   return stuff

def save(stuff, fname, **kw):
   if fname.endswith('.nc'):
      import xarray
      if not isinstance(stuff, xarray.Dataset):
         raise ValueError('can only save xarray.Dataset as .nc file')
      stuff.to_netcdf(fname)
   elif fname.count('.') == 0 or is_pickle_fname(fname):
      save_pickle(stuff, fname, **kw)
   elif fname.endswith('.json'):
      with open(fname, 'w') as out:
         out.write(json.dumps(stuff, sort_keys=True, indent=4))
   elif fname.endswith('.json.xz'):
      jsonstr = json.dumps(stuff, sort_keys=True, indent=4)
      with lzma.open(fname, 'wb') as out:
         out.write(jsonstr.encode())
   else:
      raise ValueError('dont know now to handle file ' + fname)

def save_pickle(stuff, fname, add_dotpickle=True, uselzma=False, **kw):
   opener = open
   if isinstance(stuff, wu.Bunch):
      # pickle as dict to avoid version problems or whatever
      _special = stuff._special
      stuff = dict(stuff)
      stuff['__I_WAS_A_BUNCH_AND_THIS_IS_MY_SPECIAL_STUFF__'] = _special
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
   if not os.path.basename(fname).count('.'):
      fname += '.pickle'
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
