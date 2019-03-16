# pyvan
PyVAN is a Python 2.7 package for assessing stellar flare candidate light-curves, especially suited to irregularly sampled light-curves of ground based astronomical surveys. The software takes a set of light-curve templates corresponding to both a type of variable you want to identify and to common contaminants. It then searches each candidate light-curve over a range of plausible template paramaters for the fit of each template most likely to have produced the observations. By comparing the likelihoods of these best-fits with one another, users can identify the light-curves that are both most likely to result from the astrophysical phenomenon of interest, and least likely to result from contaminants. By default, the software fits for flare, RR Lyrae, and quiet templates, and was created for identification of flares. However, additional functions allow capabilities for any templates that might interest you --- the docstrings for the most important (IMO) functions are pretty thorough and should help in this respect. However, I'll have some relevant examples added ASAP.

A plot of the differences of log-likelihoods for fits of the three previously mentioned templates to flare and contaminant data simulated by inducing ground-based survey quality and sampling in Kepler light-curves of known identity:

![](images/sim_scatters.png)
The black dashed line denotes the threshold above which < 1% of any contaminant population is included, retaining ~66% of all simulated flare light-curves containing an event. The corresponding paper has been submitted, and is pending review --  but can be viewed online: https://arxiv.org/abs/1903.03240

[Template fitting in action](ANIMS.md)

A quick use example for the software (until I can get more proper examples online):

--------------------------------------------------------------------

import pickle \
import numpy as np \
import pyvan.pyvan as pyvan \

lightcurves = pickle.load(open('lightcurves.p', 'rb' ) )\
#a pickled list where each entry is a numpy structured array with column keys: 'mjd', 'mag', 'magErr' (time, magnitude, and mag error)

tar_fits = pyvan.fit(lightcurves, n_cores=3, filt='g')\
#fits all entries in 'lightcurves' for default templates using 3 processor cores and g-band filters where applicable (RR Lyrae in this case) \
#This results in a dictionary containing an entry for each target. 

--------------------------------------------------------------------

Each target's entry in the dictionary resulting above contains information regarding each of its fits (i.e. tar_fits[0]['flare'] contains information for the 1st light-curve's flare fit) and the comparison metrics for those fits (tar_fits[0]['rel_fit'], where flare-quiescent is the difference of the best-fit log likelihoods for those templates). Happy to help if anyone has questions.
