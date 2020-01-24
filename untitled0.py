#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:41:18 2020

@author: beli
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scig
import scipy.optimize as scopt

def Gauss(x,a,b,c,d):
    return a*np.exp(-(x-b)**2/(2*c**2)) + d

def root(x,a,b,c):
    return -a*(-x+b)**(1/2) + c

def lin(x,a,b):
    return a*x + b

def quad(x,a,b,c):
    return a*(x-b)**2 + c

#Data import
data_BaGr = pd.read_csv('20200110 BaGr 600s.txt', header = 1)
data_Mix = pd.read_csv('20200110 Mix 600s.txt', header = 1)
data_Na = pd.read_csv('20200110 Na-22 600s.txt', header = 1)

#Uncalibrated E, peak find for calibration
uE_Na = data_Na['E'][:len(data_BaGr['E'])-2]
n_Na = ((data_Na['N']) [:len(data_BaGr['E'])-2]
        -(data_BaGr['N']) [:len(data_BaGr['E'])-2]) #delete background
dn_Na = (n_Na)**(1/2)
fit_b = 170
fit_e = 260
uxfit = np.linspace(fit_b,fit_e,200,endpoint=True)

upoptNa,upcovNa = scopt.curve_fit(Gauss,uE_Na[fit_b:fit_e],n_Na[fit_b:fit_e],
                                  p0 = (17,210,5,1), 
                                       sigma = (dn_Na/600)[fit_b:fit_e])
uperrNa = np.sqrt(np.diag(upcovNa))

E_Na = uE_Na * 511 / upoptNa[1] #calibration fix
xfit = uxfit * 511 / upoptNa[1]

plt.rcParams['axes.formatter.use_locale'] = True

plt.figure(0)
plt.errorbar(uE_Na, (n_Na/600), yerr = (dn_Na/600),
             ls= 'None', label = r'Uncalibrated $^{22}\mathrm{Na}$',
             marker = 'o',markeredgecolor='black', 
             capsize=5, ecolor = 'black', color = 'blue',
             zorder = 5)
plt.axvline(upoptNa[1], ls = '--', color = 'r', 
            label = '$E_{^{22}\mathrm{Na}}$ = %.1f' % 
            upoptNa[1], zorder = 10)
plt.grid(zorder = 0)
plt.legend()
plt.xlabel(r'uncalibrated $E$ (-)')
plt.ylabel(r'Vervalsnelheid $n$ ($\mathrm{s}^{-1}$)')            
            
plt.figure(1)
plt.errorbar(uE_Na[fit_b:fit_e], (n_Na/600)[fit_b:fit_e],
             yerr = (dn_Na/600)[fit_b:fit_e],
             ls= 'None', label = r'Uncalibrated $^{22}\mathrm{Na}$',
             marker = 'o',markeredgecolor='black', 
             capsize=5, ecolor = 'black', color = 'blue',
             zorder = 5) 
plt.plot(uxfit,Gauss(uxfit,*upoptNa)/600,'-r', zorder = 10,
         label = r'$%.0f \cdot exp(\frac{-(x-%.1f)^2}{2%.1f^2}) + %.0f$' % 
                                  tuple(upoptNa))
plt.plot(uxfit,Gauss(uxfit,upoptNa[0]+3*uperrNa[0],upoptNa[1],
                     upoptNa[2]+3*uperrNa[2],upoptNa[3]+3*uperrNa[3])/600,
         ls = '--', color = 'gray', zorder = 20, label = r'$\sigma 3$')
plt.plot(uxfit,Gauss(uxfit,upoptNa[0]-3*uperrNa[0],upoptNa[1],
                     upoptNa[2]-3*uperrNa[2],upoptNa[3]-3*uperrNa[3])/600,
         ls = '--', color = 'gray', zorder = 20)

plt.grid(zorder = 0)
plt.legend()
plt.xlabel(r'uncalibrated $E$ (-)')
plt.ylabel(r'Vervalsnelheid $n$ ($\mathrm{s}^{-1}$)')

E_BaGr = data_BaGr['E'][:len(data_BaGr['E'])-2] * 511 / upoptNa[1]
n_BaGr = (data_BaGr['N']) [:len(data_BaGr['E'])-2]
dn_BaGr = (n_BaGr)**(1/2)
E_Mix = data_Mix['E'][:len(data_BaGr['E'])-2] * 511 / upoptNa[1]
n_Mix = ( data_Mix['N']) [:len(data_BaGr['E'])-2] - n_BaGr
dn_Mix = (n_Mix)**(1/2) + dn_BaGr
E_Na = data_Na['E'][:len(data_BaGr['E'])-2] * 511 / upoptNa[1]
n_Na = (data_Na['N']) [:len(data_BaGr['E'])-2] - n_BaGr
dn_Na = (n_Na)**(1/2) + dn_BaGr


plt.figure(2)



plt.errorbar(E_Mix, n_Mix/600, yerr = dn_Mix/600,
                     ls= 'None', label = 'Nuclide Mix',
                     marker = 'o',markeredgecolor='black', 
                     capsize=5, ecolor = 'black', color = 'blue',
                     zorder = 5) 

#plt.errorbar(E_Na, n_Na/600, yerr = dn_Na/600,
#                     ls= 'None', label = r'$^{22}\mathrm{Na}$',
#                     marker = 'o',markeredgecolor='black', 
#                     capsize=5, ecolor = 'black', color = 'red',
#                     zorder = 10) 
"""
Compton filter
"""
froot_b = 10
froot_e = 175

xrootfit = np.linspace(E_Mix[froot_b],E_Mix[froot_e], 
                       len(E_Mix[froot_b:froot_e]))

flinh_b = 176
flinh_e = 210

xlinfit_h = np.linspace(E_Mix[flinh_b],E_Mix[flinh_e],
                        len(E_Mix[flinh_b:flinh_e]))

flinl_b = 0
flinl_e = 9

xlinfit_l = np.linspace(E_Mix[flinl_b],E_Mix[flinl_e],
                        len(E_Mix[flinl_b:flinl_e]))

root_minus = root(xrootfit,0.1,E_Mix[froot_e],10.5) 
lin_high = quad(xlinfit_h, -0.0015,420,10.5)
lin_low = lin(xlinfit_l,.39,0)
x_append = np.append(xlinfit_l,np.append(xrootfit,xlinfit_h))
minusappend = np.append(lin_low,np.append(root_minus,lin_high))

plt.plot(x_append,minusappend, '-r', zorder=10, label = 'Compton filter')
plt.grid(zorder = 0)
plt.legend()
plt.xlabel(r'Energie $E$ (keV)')
plt.ylabel(r'Vervalsnelheid $n$ ($\mathrm{s}^{-1}$)')

#
#
plotminus = np.append(minusappend,(np.zeros(len(n_Mix)-flinh_e+2)))

"""
Gaussfits for data
"""
fit_gauss_b = 232
fit_gauss_e = 320

"""
Cs-137 fit
"""
x_gaussbig = np.linspace(E_Mix[fit_gauss_b],
                         E_Mix[fit_gauss_e],100)

n_Mix_filt = n_Mix/600 - plotminus

popt_Mixh, pcov_Mixh = scopt.curve_fit(Gauss,E_Mix[fit_gauss_b:fit_gauss_e],
                                       n_Mix_filt[fit_gauss_b:fit_gauss_e],
                                       p0 = (28,650,30,0), 
                                sigma = (dn_Mix/600)[fit_gauss_b:fit_gauss_e])
perr_Mixh = np.sqrt(np.diag(pcov_Mixh))

"""
Am-241 fit
"""
fit_gausl_b = 21
fit_gausl_e = 46
popt_Mixl, pcov_Mixl = scopt.curve_fit(Gauss, E_Mix[fit_gausl_b:fit_gausl_e],
                                       n_Mix_filt[fit_gausl_b:fit_gausl_e],
                             sigma = (dn_Mix/600)[fit_gausl_b:fit_gausl_e],
                                     p0 = (5,72,10,0))
perr_Mixl = np.sqrt(np.diag(pcov_Mixl))

x_gauss_lo = np.linspace(E_Mix[fit_gausl_b],
                         E_Mix[fit_gausl_e],100)

plt.figure(3)
plt.errorbar(E_Mix, n_Mix_filt, yerr = dn_Mix/600,
                     ls= 'None', label = 'Nuclide Mix',
                     marker = 'o',markeredgecolor='black', 
                     capsize=5, ecolor = 'black', color = 'blue',
                     zorder = 5) 
plt.axvline(popt_Mixh[1], ls = '--', color = 'r', 
            label = '$E_{^{137}\mathrm{Cs}}$ = %.1f keV' % 
            popt_Mixh[1], zorder = 10)
plt.axvline(popt_Mixl[1], ls = '--', color = 'green', 
            label = '$E_{^{241}\mathrm{Na}}$ = %.1f keV' % 
            popt_Mixl[1], zorder = 10)
plt.grid(zorder = 0)
plt.legend()
plt.xlabel(r'Energie $E$ (keV)')
plt.ylabel(r'Vervalsnelheid $n$ ($\mathrm{s}^{-1}$)')

"""
Cs-137 plot
"""
plt.figure(4)
plt.errorbar(E_Mix[fit_gauss_b:fit_gauss_e], 
             n_Mix_filt[fit_gauss_b:fit_gauss_e], 
                       yerr = (dn_Mix/600)[fit_gauss_b:fit_gauss_e],
                     ls= 'None', label = r'$^{137}\mathrm{Cs}$ piek',
                     marker = 'o',markeredgecolor='black', 
                     capsize=5, ecolor = 'black', color = 'blue',
                     zorder = 5) 
plt.plot(x_gaussbig,Gauss(x_gaussbig,*popt_Mixh),'-r', zorder = 10,
         label = r'$%.1f \cdot exp(\frac{-(x-%.1f)^2}{2%.1f^2}) + %.1f$' % 
                                  tuple(popt_Mixh))
plt.plot(x_gaussbig,Gauss(x_gaussbig,popt_Mixh[0]+3*perr_Mixh[0],popt_Mixh[1],
                     popt_Mixh[2]+3*perr_Mixh[2],popt_Mixh[3]+3*perr_Mixh[3]),
         ls = '--', color = 'gray', zorder = 20, label = r'$\sigma 3$')
plt.plot(x_gaussbig,Gauss(x_gaussbig,popt_Mixh[0]-3*perr_Mixh[0],popt_Mixh[1],
                     popt_Mixh[2]-3*perr_Mixh[2],popt_Mixh[3]-3*perr_Mixh[3]),
         ls = '--', color = 'gray', zorder = 20)
plt.grid(zorder = 0)
plt.legend()
plt.xlabel(r'Energie $E$ (keV)')
plt.ylabel(r'Vervalsnelheid $n$ ($\mathrm{s}^{-1}$)')

"""
Am-241 plot
"""
plt.figure(5)
plt.errorbar(E_Mix[fit_gausl_b:fit_gausl_e], 
             n_Mix_filt[fit_gausl_b:fit_gausl_e], 
                       yerr = (dn_Mix/600)[fit_gausl_b:fit_gausl_e],
                     ls= 'None', label = r'$^{241}\mathrm{Am}$ piek',
                     marker = 'o',markeredgecolor='black', 
                     capsize=5, ecolor = 'black', color = 'blue',
                     zorder = 5) 
plt.plot(x_gauss_lo,Gauss(x_gauss_lo,*popt_Mixl),'-r', zorder = 10,
         label = r'$%.1f \cdot exp(\frac{-(x-%.1f)^2}{2%.1f^2}) + %.1f$' % 
                                  tuple(popt_Mixl))
plt.plot(x_gauss_lo,Gauss(x_gauss_lo,popt_Mixl[0]+3*perr_Mixl[0],popt_Mixl[1],
                     popt_Mixl[2]+3*perr_Mixl[2],popt_Mixl[3]+3*perr_Mixl[3]),
         ls = '--', color = 'gray', zorder = 20, label = r'$\sigma 3$')
plt.plot(x_gauss_lo,Gauss(x_gauss_lo,popt_Mixl[0]-3*perr_Mixl[0],popt_Mixl[1],
                     popt_Mixl[2]-3*perr_Mixl[2],popt_Mixl[3]-3*perr_Mixl[3]),
         ls = '--', color = 'gray', zorder = 20)
plt.grid(zorder = 0)
plt.legend()
plt.xlabel(r'Energie $E$ (keV)')
plt.ylabel(r'Vervalsnelheid $n$ ($\mathrm{s}^{-1}$)')

"""
Calculations
"""


A_Cs = (popt_Mixh[0]+popt_Mixh[3])/.851
A_Am = (popt_Mixl[0]+popt_Mixl[3])/.359

ratio = A_Cs / A_Am

dCs = popt_Mixh[1] / upoptNa[1] * uperrNa[1] + perr_Mixh[1]
dAm = popt_Mixl[1] / upoptNa[1] * uperrNa[1] + perr_Mixl[1]

dRa =( (perr_Mixh[0]+perr_Mixh[2]) / (popt_Mixl[0]+popt_Mixl[3]) * .359/.851 +
      ratio/(popt_Mixl[0]+popt_Mixl[3]) * (perr_Mixl[0]+perr_Mixl[3]))
      

