import numpy as np
import willutil as wu

class MonteCarlo:
   def __init__(
      self,
      scorefunc=None,
      temperature=1,
      debug=False,
      timer=None,
      **kw,
   ):
      self.best = 9e9
      self.bestconfig = None
      self.low = 9e9
      self.lowconfig = None
      self.last = 9e9
      self.naccept = 0
      self.ntrials = 0
      self.temperature = temperature
      self.accepted_last = False
      self.new_best_last = False
      self.scorefunc = scorefunc
      self.startconfig = None
      self.debug = debug
      self.timer = wu.Timer() if timer is None else timer

   @property
   def acceptfrac(self):
      return self.naccept / self.ntrials

   def try_this(
      self,
      config=None,
      score=None,
      **kw,
   ):
      assert score is not None or self.scorefunc is not None
      self.timer.checkpoint('try_this_begin')
      if score is None:
         self.timer.checkpoint('trythis')
         score = self.scorefunc(config, **kw)
         if self.debug: print('trythis trial score', score)
         self.timer.checkpoint('trythis score')
      if self.startconfig is None:
         self.startconfig = config
      self.last = score
      self.accepted_last = False
      self.new_best_last = False
      self.ntrials += 1
      # if self.debug: print(score)
      delta = score - self.low
      if score > 1_000_000:
         ic(f'WARNING MonteCarlo.try_this score {score} is pretty high....')
      if np.exp(max(-99, min(99, -delta / self.temperature))) > np.random.rand():
         self.naccept += 1
         self.accepted_last = True
         self.low = score
         self.lowconfig = config
         # ic('accept', score - self.best)

         if self.debug: print('trythis accept', score)
         if self.low < self.best:
            # ic('!!!!!!!!!!!!!!!!!!!!!!! mc best !!!!!!!!!!!!!!!!!!!!!')
            # ic(config)
            self.best = score
            self.bestconfig = config.copy()
            self.new_best_last = True

            if self.debug: print('trythis new best', score)
      self.timer.checkpoint('try_this_end')
      return self.accepted_last
