from __future__ import print_function

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
import astropy.units as u
import scipy.special as special
import scipy.stats as stats

import astropy.coordinates as coord


"""
Mahalanobis distance

Returns the Mahalanobis distance from a point index ih of an (N-dimensional) array Xall, with inverted covariance matrix Cinv, to the (Ndense)th nearest neighbour
"""
def Mahalanobis_distance(Xall,Cinv, ih, Ndense=20):
	dall = np.zeros(Xall.shape[1])
	for iden in range(len(dall)):
		dall[iden] = distance.mahalanobis(Xall.T[ih],Xall.T[iden],  Cinv)
	dN =np.sort(dall)[Ndense]
	return dN


"""
gauss

Return values of a gaussian function (val) given a mean, variance, and weight (normalisation)
"""
def gauss(val, mean, var, weight):
	std = np.sqrt(var)
	exp_comp = np.exp(-0.5*((val-mean)/std)**2)
	return weight*exp_comp/(2.*np.sqrt(np.pi)*std)


"""
gauss_fit

Fit a multi (between Nmin and Nmax) component Gaussian to a given distribution of points (data)
Outputs the model parameter
"""
def gauss_fit(data, Nmin=2, Nmax=2):
	N = np.arange(Nmin, Nmax+1)
	models = [None for i in range(len(N))]

	for i in range(len(N)):
		models[i] = GaussianMixture(N[i]).fit(data.reshape(-1,1))

	# compute the AIC and the BIC
	AIC = [m.aic(data.reshape(-1,1)) for m in models]
	BIC = [m.bic(data.reshape(-1,1)) for m in models]
	mod_h = models[np.argmin(AIC)]

	ms_h = mod_h.means_[:,0]
	covs_h = mod_h.covariances_[:,0,0]
	weights_h = mod_h.weights_[:]
	covs_h=  covs_h[np.argsort(ms_h)]
	ms_h = ms_h[np.argsort(ms_h)]
	weights_h = weights_h[np.argsort(ms_h)]
	
	return mod_h, AIC, BIC, ms_h, covs_h, weights_h, np.argsort(ms_h)


"""
density

Calculate the relative phase space densities of stars from the gaia catalogue
Arguments should include parallax, ra, dec, proper motions in ra and dec - all with units, plus vrad if 6D. 
Can also contain the index of the 'host star' (ih), the number of neighbours used for the density calculation (Ndense), the number of stars for which to calculate the density (NST), a pre-calculated covariance matrix of appropriate dimension, a subset of stars for which to calculate the densiies (subarr - replaces NST), a radius of a sphere in which to draw stars on which to perform the density calculations (subsize), flag to perform the calculations on the velocities only (Vonly), and the minimum number of astrometric datapoints (stars) to perform the analysis (Nmin).
"""
def density(pllx, ra, dec, pmra, pmdec, ih=None, Ndense=20, NST=1,   Cmat=None, vrad=None, subarr=None, subsize=None, Vonly=False, Nmin=50):

	dense = np.zeros(NST)
	dist = pllx.to(u.pc, equivalencies=u.parallax())
	c = coord.ICRS(ra=ra, dec=dec,  distance=pllx.to(u.pc, u.parallax()))
	gc= c.transform_to(coord.Galactocentric)
	x = gc.x
	y = gc.y
	z = gc.z

	ih_ss = ih
	if type(vrad)==type(None):
		Xall = np.array([x, y, z, pmra/pllx,pmdec/pllx])
		if len(x)<Nmin:
			print('Insufficient astrometric datapoints (Nmin = %d):'%(Nmin), len(x))
			return None, None, None, None, None
		if Vonly:
			Xall = np.array([pmra/pllx,pmdec/pllx])		
	else:
		vr_inds = np.where(~np.isnan(vrad))[0]
		if type(ih)!=type(None):
			if not ih in vr_inds or len(vr_inds)<2*Ndense:
				print('No radial velocity measurement for stellar host:', ih)
				return None, None, None, None, None
			ih_ss = np.where(vr_inds==ih)[0]
		elif len(vr_inds)<Nmin:
			print('Insufficient stars with radial velocities (Nmin = %d):'%(Nmin), len(vr_inds))
			return None, None, None, None, None
			
		x = x[vr_inds]
		y = y[vr_inds]
		z = z[vr_inds]
		ra = ra[vr_inds]
		dec= dec[vr_inds]
		dist = dist[vr_inds]
		pmra = pmra[vr_inds]
		pmdec = pmdec[vr_inds]
		pllx = pllx[vr_inds]
		vrad = vrad[vr_inds]
		
		
		c = coord.ICRS(ra=ra, dec=dec,  distance=pllx.to(u.pc, u.parallax()),   pm_ra_cosdec=pmra, pm_dec=pmdec, radial_velocity=vrad)
		gc= c.transform_to(coord.Galactocentric)
		x = gc.x
		y = gc.y
		z = gc.z
		vx = gc.v_x
		vy = gc.v_y
		vz = gc.v_z

		Xall = np.array([x, y, z,vx, vy,vz])
		if Vonly:
			Xall = np.array([vx, vy,vz])
			


	Ndims = Xall.shape[0]
	if type(Cmat)==type(None):
		Cmat = np.cov(Xall)
	Cinv = np.linalg.inv(Cmat) 
	
	if type(ih)!=type(None):
		dN =  Mahalanobis_distance(Xall,  Cinv, ih_ss, Ndense=Ndense)
		dense_host= float(Ndense)/(dN**Ndims)
		xh = x[ih_ss]
		yh = y[ih_ss]
		zh = z[ih_ss]
	else:
		dense_host= None
		xh = np.median(x)
		yh = np.median(y)
		zh = np.median(z)
	

	if type(subsize)!=type(None):
		xtmp =(x-xh)/u.pc
		ytmp =(y-yh)/u.pc
		ztmp = (z-zh)/u.pc
		drhost2 = (xtmp*xtmp+ytmp*ytmp+ztmp*ztmp)
		isub_init= np.where(drhost2<subsize*subsize)[0]
	else:
		isub_init = np.arange(len(x))
	
	if type(subarr)!=type(None):
		NST = len(subarr)
		dense= np.zeros(NST)
		isub = subarr
	elif NST<len(isub_init):
		dense = np.zeros(NST)
		isub = np.random.choice(isub_init, size=NST, replace=False)
	else:
		isub =isub_init
		dense = np.zeros(len(isub_init))

	ict=0
	for ist in isub:
		dN = Mahalanobis_distance(Xall,  Cinv, ist, Ndense=Ndense)
		dense[ict]= float(Ndense)/(dN**Ndims)
		ict+=1

	norm_dense = np.median(dense)
	dense /= norm_dense
	
	if type(dense_host)!=type(None):
		dense_host /= norm_dense

	if type(vrad)!=type(None):
		isub = vr_inds[isub]

	if plot:
		if not type(hostname)==type(None):
			hostn = hostname
		else:
			hostn = ''
		prob_clump, Cmet, Pnull, model =  analyse_densedist(dense,dense_host, plot=False, sid=sid, hostn=hostn, show=False)
		plotf.plot_dhist(dense,Cmet, model, prob_clump=prob_clump, prob_null=Pnull, dense_host=dense_host, hostn = hostn, sid=sid, show=show)
		plotf.plot_spatial(pllx, ra, dec, pmra, pmdec, isub, dense, dense_host, ih, show=show, sid=sid, hostn=hostn)

	return dense_host, norm_dense, dense, isub, Cmat
	
	
"""
analyse_densedist
Input: dense_orig = array of phase space densities
dense_host = exoplanet host phase space density (or None)

Output: if dense_host is not None, returns Phigh of the host and Pnull that the distribution of densities is lognormally distributed 
Otherwise an array with all probablities from dense_orig are returned
"""
def analyse_densedist(dense_orig, dense_host=None, dfilt_th=2.0,modparams=False):
	dfilt = np.where((dense_orig>np.percentile(dense_orig,dfilt_th))&(dense_orig<np.percentile(dense_orig,100.-dfilt_th)))[0]
	dense = dense_orig[dfilt]
	
	norm_dense=np.median(dense)
	dense /= norm_dense	
	dense_orig/=norm_dense
	mod_h, AIC, BIC, ms_h, covs_h, weights_h, isrt_h= gauss_fit(np.log10(dense), Nmin=2, Nmax=2)
	

	if type(dense_host)!=type(None):
		mod_n, AIC_n, BIC_n, ms_n, covs_n, weights_n, isrt_n = gauss_fit(np.log10(dense), Nmin=1, Nmax=1)
		
		def cdf(d):
			xtmp = (np.log10(d)-ms_n[0])/(np.sqrt(2.)*np.sqrt(covs_n[0]))
			erf = special.erf(xtmp)
			return 0.5*(1.+erf)

		
		nres =  stats.kstest(dense_orig, cdf, args=(), N=20, alternative='two-sided', mode='approx')
		Pnull = nres[1]

		isrt  = np.argsort(ms_h)
		
		prob1 = gauss(np.log10(dense_host/norm_dense), ms_h[isrt[0]], covs_h[isrt[0]], weights_h[isrt[0]])
		prob2 = gauss(np.log10(dense_host/norm_dense), ms_h[isrt[1]], covs_h[isrt[1]], weights_h[isrt[1]])

		
		Phigh =  prob2/(prob2+prob1)
		Plow =  prob1/(prob2+prob1)

	else:
		isrt  = np.argsort(ms_h)
		
		prob1 = gauss(np.log10(dense_orig), ms_h[isrt[0]], covs_h[isrt[0]], weights_h[isrt[0]])
		prob2 = gauss(np.log10(dense_orig), ms_h[isrt[1]], covs_h[isrt[1]], weights_h[isrt[1]])

		
		#Best single lognormal fit to the density distribution
		mod_n, AIC_n, BIC_n, ms_n, covs_n, weights_n, isrt_n = gauss_fit(np.log10(dense), Nmin=1, Nmax=1)
		
		def cdf(d):
			xtmp = (np.log10(d)-ms_n[0])/(np.sqrt(2.)*np.sqrt(covs_n[0]))
			erf = special.erf(xtmp)
			return 0.5*(1.+erf)
		
		Phigh =  prob2/(prob2+prob1)
		Plow =  prob1/(prob2+prob1)

		#Return KS test that the density distribution follows a lognormal function
		nres =  stats.kstest(dense, cdf, args=(), N=20, alternative='two-sided', mode='approx')
		
		#Assign Pnull
		Pnull = nres[1]
			
	#Whether to return the parameters of the Gaussian mixture models
	if not modparams:
		return Phigh, Pnull, mod_h, isrt_h
	else:
		return Phigh, Pnull, mod_h, AIC, BIC, ms_h, covs_h, weights_h

