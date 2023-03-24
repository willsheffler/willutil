import os, argparse
import numpy as np
import willutil as wu

def makesym(fname, arch, **kw):
   print('-' * 80)
   print(fname, flush=True)
   tgtaxis, nfold, frames = get_tgtaxis_frames(arch, **kw)
   pdb = wu.readpdb(fname, removehet=True)
   componentinfo = pdb.syminfo(**kw)
   pdbasu = build_component_asu(componentinfo, **kw)
   compaxis = wu.sym.axes(arch[:-1], nfold=componentinfo.nfold)
   pdbaln = align_to_axis(pdbasu, compaxis, componentinfo, cenaxis=tgtaxis, **kw)
   tag = os.path.basename(fname)
   dump_transformed_samples(pdbaln, compaxis, frames, tag=tag, arch=arch, **kw)
   print('-' * 80, flush=True)

def get_tgtaxis_frames(arch, output_symmetry, **kw):
   nfold, sym = int(arch[-1]), arch[:-1]
   tgtaxis = wu.sym.axes(sym, nfold=nfold)
   if output_symmetry == 'full': frames = wu.sym.frames(sym)
   elif output_symmetry == 'cyclic': frames = wu.sym.frames(nfold, axis=tgtaxis)
   elif output_symmetry == 'asym': frames = np.eye(4).reshape(1, 4, 4)
   else: raise ValueError(f'unknown output_symmetry {output_symmetry}')
   return tgtaxis, nfold, frames

def build_component_asu(componentinfo, reconcile_method='longest_chain', **kw):
   if reconcile_method != 'longest_chain':
      raise ValueError(f'only reconcile_method longest_chain is currently supported')
   ci = componentinfo
   chains = ci.chains
   xform = wu.hrot(ci.axis, ci.angle, ci.center)
   newchains = list()
   for ichain, _, _ in ci.chaingroups:
      newchains.append(chains[ichain])
   return wu.pdb.join(newchains)

def align_to_axis(pdb, axis, componentinfo, cenaxis=None, autoset_rotational_origin=True, **kw):
   xalign = wu.htrans(-componentinfo.center)  # move rotation axis to isect origin
   xalign = wu.halign(componentinfo.axis, axis) @ xalign  # align symaxis
   com = wu.hcom(pdb.coords)
   com2 = wu.hproj(axis, xalign @ com)
   xalign = wu.htrans(-com2) @ xalign  # move along symaxis so com as close as possible to origin
   if autoset_rotational_origin:
      assert cenaxis is not None
      newcom = wu.hxform(xalign, com)
      towardscom = wu.hprojperp(axis, newcom)
      towardscen = wu.hprojperp(axis, cenaxis)
      xalign = wu.halign2(axis, towardscom, axis, towardscen) @ xalign  # rotate asu com around symaxis
   pdbaln = wu.hxform(xalign, pdb)
   return pdbaln

def dump_transformed_samples(pdb, compaxis, frames, radius, angle, arch='', output_prefix='', tag='', **kw):
   for rad in np.arange(*radius):
      for ang in np.arange(*angle):
         xsamp = wu.hrot(compaxis, ang) @ wu.htrans(rad * compaxis)
         newpdb = wu.pdb.join([wu.hxform(xframe @ xsamp, pdb) for xframe in frames])
         fname = f'{output_prefix}_{arch}_{tag}_{rad:07.2f}_{np.degrees(ang):07.1f}.pdb'
         print('dumping', fname, flush=True)
         newpdb.dump(fname)

def main():
   kw = get_cli_config()
   for fname in kw.inputs:
      try:
         makesym(fname, **kw)
      except ValueError as e:
         print(f'Input {fname} failed, reason:')
         print(e, flush=True)

def get_cli_config():

   parser = argparse.ArgumentParser()
   parser.add_argument(
      'inputs',
      nargs='+',
      help='input pdb files',
   )
   parser.add_argument(
      '--architecture',
      default='oct4',
      help='symmetry desired, tet#, oct# or icos#, where # is the symmetry axis aroud which we build. '
      'for example, oct4 will place inputs around the 4fold axis',
   )
   parser.add_argument(
      '--radius',
      default=[100, 110, 1],
      type=float,
      nargs=3,
      help='radial samples [start, stop, step] inclusive',
   )
   parser.add_argument(
      '--angle',
      default=[-20, 20, 10],
      type=float,
      nargs=3,
      help='angular samples [start, stop, step] inclusive, in degrees',
   )
   parser.add_argument(
      '--output_prefix',
      default='makesym',
      help='prefix for output pdbs. can contain directories. fname will be '
      '{output_prefix}_{architecture}_{input basename}_{radius}_{angle}.pdb',
   )
   parser.add_argument(
      '--autoset_rotational_origin',
      default=True,
      type=bool,
      help='realign such that 0 rotation places chains as close as possible. sortof',
   )
   parser.add_argument(
      '--input_nfold',
      default=None,
      type=int,
      help='nfold of input structures. if not specified, will be auto-detected',
   )
   parser.add_argument(
      '--output_symmetry',
      type=string_arg_checker('full cyclic asym'),
      default='cyclic',
      help='asym: output one subunit, cyclic: output formed cyclic oligomer, full: output full cage',
   )
   parser.add_argument(
      '--template',
      default=None,
      help='template to align',
   )
   parser.add_argument(
      '--rms_tolerance',
      default=1.0,
      type=float,
      help='rms limit for "symmetric" chains',
   )
   parser.add_argument(
      '--angle_tolerance',
      default=0.5,
      type=float,
      help='max deviation from ideal angle (degrees) for "symmetric" chains',
   )
   parser.add_argument(
      '--seqmatch_tolerance',
      default=0.9,
      type=float,
      help='minimum sequence match fraction to consider chains "symmetrical"',
   )
   parser.add_argument(
      '--translation_tolerance',
      default=0.5,
      type=float,
      help='max allowed shift along sym axis for "symmetric" chains',
   )
   parser.add_argument(
      '--reconcile_method',
      default='longest_chain',
      type=string_arg_checker('unchanged longest_chain shortest_chain common average'),
      help='''specifies how asymmetrical chains are handled. choices:
      unchanged: align to symmetry axis but do not symmetrize structure
      longest_chain: symmetrize by copying longest chain
      shortest_chain: symmetrize by copying shortest chain 
      common: symmetrize by copying longest common sequence
      average: symmetrize by averaging longest common sequence
      ''',
   )
   args = parser.parse_args()
   args = wu.Bunch(args)

   args.radius[1] += 0.001
   args.angle = [np.radians(x) for x in args.angle]
   args.angle[1] += 0.001

   args.tolerances = wu.Bunch(
      rms=args.rms_tolerance,
      translation=args.translation_tolerance,
      angle=np.radians(args.angle_tolerance),
      seqmatch=args.seqmatch_tolerance,
   )
   args.arch = args.architecture

   ic(args.tolerances)

   assert not args.template

   return args

def string_arg_checker(vals):
   def func(arg):
      if arg not in vals.split():
         raise argparse.ArgumentTypeError(f'invalid argument {arg}')
      return arg

   return func

if __name__ == '__main__':
   main()