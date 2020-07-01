# astrophasesplit
Decompose a local astrometric phase space distribution for a stellar population into 'high-density' and 'low-density' components.

Prerequisites: Python, numpy, scipy, sklearn and astropy

No installation required, code consists of a single Python file.

Use the 'density' function the calculate the Mahalanobis density for each star in a subset in a given neighbourhood with astrometric data (e.g. from Gaia DR2).

Use the 'analyse_densedist' function to decompose the distribution into two lognormals, and estimate the probability that each star belongs to the 'low' or 'high' density component. 
