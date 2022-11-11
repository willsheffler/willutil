import torch
import willutil as wu


def main():

    nofit = wu.load('/home/sheffler/src/BFF/rf_diffusion/will/i235_240_50_LGL_nosymfit_E_rmsvals.pickle')
    fit = wu.load('/home/sheffler/src/BFF/rf_diffusion/will/i235_240_50_LGL_symfit_E_rmsvals.pickle')

    fitm1 = wu.load('/home/sheffler/src/BFF/rf_diffusion/i235_240_50_LGL_nosymfit_E_rmsvals.pickle')

    fit = torch.tensor(fit.rmsvals)
    fit = torch.mean(fit, axis=0)[1:]

    fitm1 = torch.tensor(fitm1.rmsvals[:-1])
    fitm1 = torch.mean(fitm1, axis=0)[1:]

    nofit = torch.tensor(nofit.rmsvals)
    nofit = torch.mean(nofit, axis=0)[1:]

    # ic(nofit)

    fit = fitm1

    print(nofit - fit)
    print(torch.mean(nofit - fit))


if __name__ == '__main__':
    main()