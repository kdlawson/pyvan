# Animations

A few quick animations showing the differential evolution template fitting process for data of simulated PTF quality. In both cases, 
the purple observations are the data being fit to, while the black points represent the original Kepler light-curve from which the simulated observations were derived (the optizimation is completely unaware of the black observations). The lower panel(s) simply show a cropped in time axis to show a bit more detail. 

## Flare star light-curve with three clear events:
![](images/flare_fitting.gif)
(This is definitely an outlier in terms of data quality--- I chose to use it here because the fit is a little bit more complicated with so many flares. There is normally quite a bit more ambiguity here)

## RR Lyrae recovery with only a dozen or so observations:
![](images/rrl_fitting.gif)
(This procedure is a little bit different than the one we actually use. Here, I've set the 23 RR Lyrae templates themselves as a free parameter for optimization. Our normal technique tends to converge a bit faster, but is decidedly more of a pain to show this way)