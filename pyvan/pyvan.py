#!/usr/bin/env python

# https://github.com/kdlawson/pyvan
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import lmfit
import warnings
import pickle
import glob
import pkg_resources
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from copy import deepcopy
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde
from matplotlib.widgets import Button

warnings.filterwarnings(action='ignore', category=UserWarning, module='lmfit')
# Suppresses an erroneous UserWarning from LMFit indicating that a keyword argument is being ignored when it isn't
NoneType = type(None)


def med_abs_dev(arr):
    """
    Computes the scaled median absolute deviation of an array. This is a measure of sample variance robust to outlier
    events.
    """
    return np.median(abs(arr - np.median(arr))) / norm.ppf(3 / 4.)


def get_cands(data):
    """
    Default candidacy test for flares used in Lawson+(2019). Requires a flare event candidate have two consecutive
    bright outlier observations, one of which is an outlier of 5+ median absolute deviations, and the second of which
    is an outlier of at least 2.5 median absolute deviations.
    """
    flare_cands = []
    thresh1 = -5.0  # Sigma Threshold required for first candidate point
    thresh2 = -2.5  # Sigma Threshold required for second candidate point
    upper_tlimit = 4. * 100. / 60. / 24.

    med = np.median(data['mag'])
    madev = med_abs_dev(data['mag'])
    mfm_series = (data['mag'] - med) / madev  # Number of median absolute deviations of each point from the median

    bright_outliers = np.where(mfm_series <= thresh1)[0]
    bright_outliers = np.array(bright_outliers)[
        np.argsort(mfm_series[bright_outliers])]  # Sorts in order of MAD from median

    for i in range(len(bright_outliers)):
        f_index = bright_outliers[i]
        if f_index != 0:
            dt = data[f_index]["mjd"] - data[f_index - 1]["mjd"]
            if mfm_series[f_index - 1] <= thresh2 and dt < upper_tlimit and (f_index - 1) not in flare_cands:
                flare_cands = np.append(flare_cands, [f_index])
        if f_index < len(data) - 1:
            dt = data[f_index + 1]["mjd"] - data[f_index]["mjd"]
            if mfm_series[f_index + 1] <= thresh2 and dt < upper_tlimit and (
                    f_index + 1) not in flare_cands and f_index not in flare_cands:
                flare_cands = np.append(flare_cands, [f_index])

    flare_cands = np.array(flare_cands).astype(int)
    if len(flare_cands) == 0:
        return flare_cands
    peaks = np.array(find_apparent_peak_indices(data, flare_cands))
    return peaks[np.argsort(data['mag'][peaks])]


def _default_failure_threshold(target_fits):
    """
    Default threshold function for truncating fitting on a target. Value of dl_fq_threshold is based on analysis in
    Lawson+(2019). Returns True if the procedure is to skip further fits. Additionally requires that the flare fit
    completed -- i.e. that at least one flare event candidate was found according to the function supplied by
    'flare_cands'.
    """
    if 'flare' in target_fits:
        if not target_fits['flare']['fit']:
            return True  # If the flare fit has been attempted but failed (due to lack of candidates), returns True
        if 'quiescent' in target_fits:
            dl_fq_threshold = 10.44  # Determined with simulation fitting, see: Lawson+(2019)
            if target_fits['flare']['logL'] - target_fits['quiescent']['logL'] < dl_fq_threshold:
                return True
                # If the flare and quiet fit have completed but fall below threshold dl_fq value, returns True
    return False


def fit(lightcurves, n_cores, filt, threshold=_default_failure_threshold, templates=None, obj_ids=None,
        fit_dict=None, flare_cands=get_cands, expt=60.):
    """
    PyVAN's core process. This takes a list containing light-curves and fits them with templates as desired,
    distributing those light-curves among a specified number of cores, and either appending the results to a given
    dictionary or creating a new dictionary to house them.

    NOTE: If you're just fitting a single target of interest, it's probably easier to use the pyvan.fit_target
    function

    Parameters
    ----------
    lightcurves : list
        List of numpy structured arrays for each unique object to be fit with templates.

    n_cores : int
        The number of cores for PyVAN to utilize.

    filt : str
        Label for the filter to be utilized in RR Lyr template fitting. If RR Lyr fitting is not being used, "None" will
        be accepted. Default filters include: 'u', 'g', 'r', 'i', 'z'. See docstring for pyvan.make_template_dict() for
        information on adding your own filter templates. Will update soon such that 'filter' will also be passed to
        fit_general and a user defined fitting function to allow filter dependant templates to be used accordingly.

    threshold : callable, optional
        A function to which the current dictionary of fit templates is passed after each is completed, returning True if
        the procedure is to be truncated. By default, uses '_default_failure_threshold', which truncates fitting if the
        value of dl_fq is too small, or if no flare candidates were identified. This should be changed to None or a
        different user-defined function if not using flare, quiet, and at least a third template (RR Lyr by default),
        or even potentially if working with very different data quality.

    templates : list, optional
        Templates to be fit to the light-curve. Entries may be: a string for a default template ("flare", "quiescent",
        or "rrlyrae"), a dictionary for user-defined templates to be fit using PyVAN's generalized template fitting
        procedure (fit_general()), or a callable for templates to be fit according to a user-defined procedure. Note:
        if defining your own procedure, the dictionary returned must also contain a name for the function (str) under
        the key 'name'. This is the key under which the fit dictionary for that template will be stored.

    obj_ids : list, optional
        A list of unique identifiers, either strings or ints, each corresponding to the light-curve at the same
         position in "lightcurves". If 'None', obj_ids are assigned as integers beginning at the length of the current
         dictionary of fit objects, if one is provided, or zero if not.

    fit_dict : dict, optional
        If provided, new target fits are appended to fit_dict with keys determined by 'obj_ids'. If not, a new dict
        dictionary is created to fill. Alternatively, this can be used to refit a subset of objects by passing in the
        dictionary containing the targets' old fits, along with a list of their light-curves for "lightcurves" and a
        list of their identifiers as "obj_ids". The new fits will then overwrite the old entries in the dictionary,
        leaving others intact

    flare_cands : callable, optional
        A function that takes 'data' as it's argument, and returns a list of flare candidate indices for data. Set to
        'None' if not fitting for flares. If you're fitting some other template that requires specific candidate
        observations, it is recommended to use the option for passing a complete template fitting function as a template
        to PyVAN. In this function, you can then check for candidates, returning a null fit if none are identified (as
        in the procedure for flares, see: pyvan.fit_flares())

    expt : float, optional
        The exposure time in seconds for observations in the light-curve. This parameter is only used for informing the
        bounds for some templates (e.g. to avoid exploring flare fits with durations much smaller than our exposure
        time, when we require two consecutive points in the flare for candidacy by default)

    Returns
    -------
    fit_dict : dict
        Contains a key corresponding to a dictionary for each targets's fit that contained at least a single successful
        template fit (i.e. for standard flare searches, any objects that lack a single flare candidate are not
        included). See pyvan.fit_target() for detailed information regarding each target dictionary's contents.
    """
    if isinstance(templates, NoneType):
        templates = np.array(['flare', 'quiescent', 'rrlyrae'])
    if isinstance(fit_dict, NoneType):
        fit_dict = {}
    if isinstance(obj_ids, NoneType):
        obj_ids = np.arange(len(fit_dict), len(fit_dict) + len(lightcurves))
    elif len(obj_ids) != len(lightcurves):
        raise TypeError('If "obj_ids" is specified, it must be an array-like with a length equal to that of'
                        ' "lightcurves"')
    if isinstance(filt, NoneType) and 'rrlyrae' in templates:
        raise TypeError(
            "If the list of templates to be fit contains 'rrlyrae' (default), filter cannot be set to 'None'."
            " Default filter options are 'u', 'g', 'r', 'i', or 'z'")
    results = Parallel(n_jobs=n_cores)(
        delayed(fit_target)(data, filt, threshold, templates, obj_id, flare_cands, expt)
        for data, obj_id in zip(lightcurves, obj_ids))
    for target in results:
        if target['fit']:
            key = target['obj_id']
            fit_dict[key] = target
            del fit_dict[key]['obj_id']

    return fit_dict


def fit_target(data, filt, threshold=_default_failure_threshold, templates=None, obj_id=None, flare_cands=get_cands,
               expt=60.):
    """
    Carries out fitting of the requested templates on a light-curve. Use pyvan.fit() to fit many targets at once with
    multiple cores.

    Parameters
    ----------
    data : structured ndarray
        Light-curve data in structured numpy array, with columns for (at least): time, magnitude, and magnitude error.

    filt : str
        Label for the filter to be utilized in RR Lyr template fitting. If RR Lyr fitting is not being used, "None" will
        be accepted. Default filters include: 'u', 'g', 'r', 'i', 'z'. See docstring for pyvan.make_template_dict() for
        information on adding your own filter templates. Will update soon such that 'filter' will also be passed to
        fit_general and a user defined fitting function to allow filter dependant templates to be used accordingly.


    threshold : callable, optional
        A function to which the current dictionary of fit templates is passed after each is completed, returning True if
        the procedure is to be truncated. By default, uses '_default_failure_threshold', which truncates fitting if the
        value of dl_fq is too small, or if no flare candidates were identified. This should be changed to None or a
        different user-defined function if not using flare, quiet, and at least a third template (RR Lyr by default),
        or even potentially if working with very different data quality.

    templates : list, optional
        Templates to be fit to the light-curve. Entries may be: a string for a default template ("flare", "quiescent",
         or "rrlyrae"), a dictionary for user-defined templates to be fit using PyVAN's generalized template fitting
         procedure (fit_general()), or a callable for templates to be fit according to a user-defined procedure. Note:
         if defining your own procedure, the dictionary returned must also contain a name for the function (str) under
         the key 'name'. This is the key under which the fit dictionary for that template will be stored. Uses
         ["flare", "quiescent", "rrlyrae"] by default.

    obj_id : int OR string, optional
        Unique identifier corresponding to the target being fit.

    flare_cands : callable, optional
        A function that takes 'data' as it's argument, and returns a list of flare candidate indices for data. Set to
        'None' if not fitting for flares. If you're fitting some other template that requires specific candidate
        observations, it is recommended to use the option for passing a complete template fitting function as a template
        to PyVAN. In this function, you can then check for candidates, returning a null fit if none are identified (as
        in the procedure for flares, see: pyvan.fit_flares())

    expt : float, optional
        The exposure time in seconds for observations in the light-curve. This parameter is only used for informing the
        bounds for some templates (i.e. to avoid exploring flare fits with durations much smaller than our exposure
        time, when we require two consecutive points in the flare for candidacy by default

    Returns
    -------
    target_fits: dict
        Contains:
            'data', structured ndarray - a copy of the light-curve as processed

            'rel_fit', dict - contains the differences of all fit template log-likelihoods, keyed as
                'template_i-template_j'

            'expt', float - the exposure time specified by the user for the fitting

            'obj_id', int OR string - the unique identifier given for the object. This is used as the key for the target
                in the dictionary containing all of the targets that were fit, and then removed (This is done to handle
                some issues with passing certain dictionary structures in multi-core processing).

            'fit', bool - indicates whether or not the target was fit

            '<template name>', dict - a dictionary corresponding to each template. Contains at least the best-fit
                log-likelihood ('logL'), the best-fit template parameters ('params'), and whether or not the specific
                template was fit ('fit').
    """
    n_templates_fit = 0
    target_fits = {}

    if isinstance(templates, NoneType):
        templates = np.array(['flare', 'quiescent', 'rrlyrae'])

    if isinstance(filt, NoneType) and 'rrlyrae' in templates:
        raise TypeError(
            "If the list of templates to be fit contains 'rrlyrae' (default), filter cannot be set to 'None'."
            " Default filter options are 'u', 'g', 'r', 'i', or 'z'")

    threshold_fail = False
    for template, i in zip(templates, range(len(templates))):
        if threshold_fail:
            template_fit = {'fit': False, 'logL': np.nan}
            if isinstance(template, dict):
                template = templates[i] = template['name']
        elif callable(template):  # If passed a complete fitting function for a template, passes 'data' to that function
            template_fit = template(data)
            template = templates[i] = template_fit['name']
        elif isinstance(template, dict):  # If passed a dict in the templates list, uses the general fitting procedure
            template_fit = fit_general(data, template['fn'], template['bounds'], template['args'])
            template = templates[i] = template['name']
        elif template == 'flare':
            template_fit = fit_flares(data, expt=expt, flare_cands=flare_cands, N_concurrent=3)
        elif template == 'quiescent':
            template_fit = fit_quiescence(data)
        elif template == 'rrlyrae':
            template_fit = fit_rrlyrae(data, filt)
        else:
            raise TypeError('Entries of argument "templates" must either be: a string for default templates ("flare", '
                            '"quiescent", or "rrlyrae"), a dictionary for user-defined templates to be fit using PyVANs'
                            'generalized template fitting procedure, or a callable function for templates to be fit '
                            'according to a user-defined procedure.')
        if template_fit['fit']:
            n_templates_fit += 1
        target_fits[template] = template_fit
        if not isinstance(threshold, NoneType):
            threshold_fail = threshold(target_fits)
    rel_fits = {}
    for i in range(len(templates)):
        for j in range(len(templates) - (i + 1)):
            rel_fits[templates[i] + '-' + templates[j + i + 1]] = target_fits[templates[i]]['logL'] - \
                                                                  target_fits[templates[j + i + 1]]['logL']
    target_fits['data'] = data
    target_fits['rel_fit'] = rel_fits
    target_fits['expt'] = expt
    if not isinstance(obj_id, NoneType):
        target_fits['obj_id'] = obj_id
    target_fits['fit'] = (True if n_templates_fit > 0 else False)  # For dumping targets that weren't fit
    return target_fits


def fit_general(data, fn, bounds, args):
    """
    Function carrying out generalized procedure for fitting a user-provided template. NOTE: Currently, for reasons that
    are unclear to me, any functions being called within your "fn" or "bounds" functions needs to be imported within
    your functions. i.e., if your bounds are defined by "my_bounds", which makes a call somewhere to a numpy
    function, you would need to do something like:

    def my_bounds(data):
        import numpy as np
        [...]

    This is obviously not ideal. I intend to push out a better solution for this ASAP.

    Parameters
    ----------
    data : structured ndarray
        Light-curve data in structured numpy array, with columns for: time, magnitude, and magnitude error

    fn : callable
        Template function which takes arguments of time and the entries of 'args' and returns corresponding magnitudes

    bounds : callable OR list
        If callable: a function which takes 'data' as its argument and returns a list of tuples providing a lower and
        upper bound for each template parameter in 'args', in the same order as in args.
        If list: a list of tuples providing a lower and upper bound for each template parameter in 'args', in the same
        order as in args

    args : list
        A list of strings for the parameters over which 'fn' is to be fit.

    Returns
    -------
    : dict
        Dictionary containing the best-fit chi-squared, log-likelihood and params, a copy of the input fitting function,
        and 'fit'-- a boolean indicating that the fit was performed.
    """
    model = lmfit.Model(fn)
    if callable(bounds):
        arg_bounds = bounds(data)
    else:
        arg_bounds = bounds

    for arg, i in zip(args, range(len(args))):
        model.set_param_hint(arg, vary=True, value=np.mean(arg_bounds[i]), min=arg_bounds[i][0],
                             max=arg_bounds[i][1])
    result = model.fit(data['mag'], t=data['mjd'], weights=1. / data['magErr'], method='differential_evolution')
    fit_params = np.array([result.best_values[arg] for arg in args])
    chisq = result.chisqr
    return {'chisq': chisq, 'logL': log_likelihood(chisq, len(data)), 'params': fit_params, 'fit': True, 'fn': fn}


def parse_lc_file_to_list(lc_path, dtypes=None, oid_col='obj_id', time_col='mjd', return_oids=True):
    """
    Loads a large text file containing observations of many targets (probably resulting from a light-curve database
    query), sorts it in place by an integer identifier and then by observation times, and then splits it on the
    identifiers to produce a list whose length is the number of object identifiers contained in the file, and where each
    entry is the full light-curve for an object. This is by far the fastest method I've found for handling light-curve
    files like this. At a minimum, individual light-curves processed with PyVAN must contain magnitudes, magnitude
    errors, and observation times. In order to be prepared using this function, they must also have an entry providing
    a unique integer identifier for the target.

    Note: We explicitly define dtypes because numpy will otherwise assume a dtype of 'int' for the 'obj_id' column for
    light-curves from the Large Survey Database, whose entries are sometimes too long for this datatype. I don't believe
    there is currently a way to dictate only a single column's data-type when reading in a numpy structured array, so
    here we are.

    Parameters
    ----------
    lc_path : string
        The path leading to the tab-delimited file containing observations to be parsed.

    dtypes : list, optional
        A list of tuples, with one tuple corresponding to each column of the file. If not specified, assumes the format
        for PTF outputs from the Large Survey Database used in Lawson+(2019): 10 columns corresponding to RA, Dec, MJD,
        Object ID, Filter ID, magnitude, magnitude error, limiting mag, PID, and det. zero-point.

    oid_col : string, optional
        Label in 'dtypes' that corresponds to the integer identifier that delineates different targets

    time_col : string, optional
        Label in 'dtypes' that corresponds to the timestamp of the observations.

    return_oids : bool, optional
        If True, also return an array of unique object identifiers, corresponding to the list of light-curves returned.

    Returns
    -------
    lightcurves : list
        List of numpy structured arrays for each unique object in the list of lightcurves provided.
    obj_ids : ndarray
        Array containing object identifiers corresponding to the entries of 'lightcurves'. Only provided if
        'return_oids' is True.
    """
    if isinstance(dtypes, NoneType):
        dtypes = [('ra', '<f8'), ('dec', '<f8'), ('mjd', '<f8'), ('obj_id', np.uint64), ('fid', '<f8'), ('mag', '<f8'),
                  ('magErr', '<f8'), ('limMag', '<f8'), ('pid', '<i8'), ('det_zp', '<f8')]

    observations = np.genfromtxt(lc_path, dtype=dtypes, skip_header=True)
    observations.sort(order=[oid_col, time_col], axis=0)
    lightcurves = np.split(observations, np.where(np.diff(observations[oid_col]) != 0)[0] + 1)

    if return_oids:
        obj_ids = np.unique(observations[oid_col])
        return lightcurves, obj_ids

    return lightcurves


def find_apparent_t0(data, peak, m0, dm):
    """
    Used in initializing flare fit bounds for start time
    """
    max_dt = 0.16  # Very long to be thorough
    t_before, m_before = data['mjd'][:peak + 1], data['mag'][:peak + 1]
    for i in range(len(t_before)):
        dt = t_before[-1] - t_before[-(1 + i)]
        if m_before[-(1 + i)] > (m0 - 0.05 * dm - np.std(data['mag'])) and dt < max_dt:
            return t_before[-(1 + i)]
    return data['mjd'][peak] - max_dt


def find_apparent_peak_indices(data, peak_candidates):
    """
    Used to help ensure best candidates for flare fitting.
    """
    peak_candidates = np.unique(peak_candidates)
    seperate_events = np.split(peak_candidates, np.where((abs(np.diff(peak_candidates)) != 1))[0] + 1)
    peak_list = []
    for i in range(len(seperate_events)):
        event_mags = data['mag'][np.array(seperate_events[i]).astype(int)]
        peak_index = int(seperate_events[i][np.where(event_mags == np.min(event_mags))[0][0]])
        peak_list.append(peak_index)
    return peak_list


def tighten_flare_fit(input_fit_dict, copy=True):
    """
    Can be used to refit a flare template to achieve the smallest amplitude fit that achieves approximately the same
    log-likelihood.
    """
    if copy:
        fit_dict = deepcopy(input_fit_dict)
    else:
        fit_dict = input_fit_dict
    data = fit_dict['data']
    peaks = fit_dict['flare']['peaks']
    expt = fit_dict['expt'] / (3600. * 24.)
    fit_params = fit_dict['flare']['params']
    m0_init, m0_err, t0_init, dt_init, dm_init = initialize_flare_params(data, peaks)

    n = len(peaks)
    full_chisq = fit_dict['flare']['chisq']

    new_params = np.copy(fit_params)
    for i in range(n):
        wo_params = np.append(fit_params[0:i * 3 + 1], fit_params[i * 3 + 4:])
        temp_dict = {'flare': {}, 'data': data}
        temp_dict['flare']['params'] = wo_params
        _, current_model = get_flare_model(temp_dict)
        wo_chisq = chi_squared(data['mag'], current_model, data['magErr'])
        chisq_contr = wo_chisq - full_chisq
        weights = np.copy(1. / data['magErr'])
        suppressed_peaks = peaks[peaks != peaks[i]]
        weights[suppressed_peaks] = 0
        for j in np.linspace(0.0001, 5, 20):
            fmodel = lmfit.Model(_flarefn)
            fmodel.set_param_hint('t0', vary=True, value=t0_init[i], min=(t0_init[i] - 0.08),
                                  max=(t0_init[i] + dt_init[i] - expt))
            dm_min = 0.
            dm_max = ((dm_init[i] + j) if (dm_init[i] + j) > dm_min else dm_min + 0.0001)
            fmodel.set_param_hint('dm', vary=True, value=dm_init[i], min=0, max=dm_max)
            fmodel.set_param_hint('dt', vary=True, value=dt_init[i], min=expt / 5., max=0.055)
            result = fmodel.fit(data['mag'], t=data['mjd'], weights=1. / data['magErr'],
                                method='differential_evolution', model_in=current_model)
            if wo_chisq - result.chisqr >= chisq_contr * 0.999:
                new_params[i * 3 + 1:i * 3 + 4] = [result.best_values['t0'], result.best_values['dt'],
                                                   result.best_values['dm']]
                break

    fit_dict['flare']['params'] = new_params
    _, new_model = get_flare_model(fit_dict, high_res=False)
    fit_dict['flare']['chisq'] = chi_squared(data['mag'], new_model, data['magErr'])
    fit_dict['flare']['logL'] = log_likelihood(fit_dict['flare']['chisq'], len(data))
    return fit_dict


def finer_time_resolution(params, data):
    """
    Used for plotting flare fits. Attempts to create an array of time values that has as few points as possible along
    the quiescence, but enough points for a smooth function during the flare events.
    """
    params = params[1:]  # drops the quiescence for ease of iteration
    for i in range(len(params) / 3):
        k = i * 3
        t0, dt = params[k + 0], params[k + 1]
        end_time = (t0 + dt) + 30 * dt
        flare = np.concatenate((np.linspace(t0 - (0.25 * dt), t0 + dt, 200), np.linspace(t0 + dt, end_time, 800)),
                               axis=0)
        if i == 0:
            qui_before = np.linspace(np.min(data['mjd']), t0, 10, endpoint=True)
            qui_after = np.linspace(end_time, np.max(data['mjd']), 10, endpoint=True)
            t = np.concatenate((qui_before, flare, qui_after), axis=0)
        else:
            t = np.concatenate((t, flare), axis=0)
    return np.sort(np.unique(t))


def _flare_rise(T, dF):
    """
    Evaluates the rise phase for flares in fractional flux space and in rise-duration scaled times
        -- via Davenport+ (2014)

    NOTE: This is a small departure from the template as described in the text above. The paper describes the template
        as being parameterized in t_1/2 -- in units of flare FWHM. However, looking at the paper's plots, the template
        appears to be parameterized as I've described it here: with time in units of flare rise duration (hence the
        plots showing the template have a time axis in which the rise phase occurs over exactly 1 unit)

    Parameters
    ----------
    T : 1-d array-like
        The array of times in units of flare rise phase

    dF : float
        The fractional flux amplitude of the flare event. Assumes a quiet level flux of 0.

    Returns
    -------
    : 1-d array
        The fractional fluxes for the flare rise phase evaluated at T
    """

    co = [1.941, -0.175, -2.246, -1.125]
    rise = (co[0] * T + co[1] * T ** 2 + co[2] * T ** 3 + co[3] * T ** 4) + 1.
    return dF * rise


def _flare_decay(T, dF):
    """
    Evaluates the decay phase for flares in fractional flux space and in rise-duration scaled times
        -- via Davenport+ (2014)

    Parameters
    ----------
    T : 1-d array-like
        The array of times in units of flare rise phase

    dF : float
        The fractional flux amplitude of the flare event. Assumes a quiet level flux of 0.

    Returns
    -------
    : 1-d array
        The fractional fluxes for the flare rise phase evaluated at T
    """

    co = [0.6890, -1.6000, 0.3030, -0.2783]  # Davenport+ (2014) flare decay phase coefficients
    decay = co[0] * np.exp(co[1] * T) + co[2] * np.exp(co[3] * T)
    return dF * decay


def quiescence(t, qui):
    """
    Used for 'evaluating' a quiescent level at an array of time values

    Parameters
    ----------
    t : 1-d array-like
        The array of time values for which to evaluate the model

    qui : float
        The value of the quiet level (mag or flux)

    Returns
    -------
    : 1-d array
        The quiescent array corresponding to t
    """
    return np.repeat(qui, len(t))


def _flarefn(t, t0, dt, dm, **kwargs):
    """
    Used internally by the main flare fitting algorithm ('fit_flares') to iteratively
    append flares (in excess of the N_concurrent parameter in fit_flares) to the previous model,
        -- Template model described in Davenport+ (2014).

    NOTE: This is a small departure from the template as described in the text above. The paper describes the template
        as being parameterized in t_1/2 -- in units of flare FWHM. However, looking at the paper's plots, the template
        appears to be parameterized as I've described it here: with time in units of flare rise duration (hence the
        plots showing the template have a time axis in which the rise phase occurs over exactly 1 unit, but where the
        FWHM does not)

    With LMFit we're able to pass "model_in" as a keyword-arg -- an array of the current model magnitude values.
    The set of trial parameters is used to append the jth flare event to the previous composite (j-1) flare model.

    Parameters
    ----------
    t : 1-d array-like
        The array of time values for which to evaluate the flare model

    t0 : float
        The beginning of the flare event's rise phase

    dt : float
        The duration of the flare's rise phase

    dm : float
        The amplitude of the flare event in magnitudes

    **kwargs : Accepts 'model_in' or 'm0'
        - model_in : 1-d array-like
            An array of magnitudes onto which the flare will be added.
        - m0 : float
            If model_in is not specified, the value of m0 will be used as the quiescent level instead of 0.

    Returns
    -------
    model_out : 1-d array
        The magnitude array of the composite flare model evaluated at t
    """

    T = (t - t0 - dt) / dt

    model_out = np.zeros((2, len(T)))

    if 'model_in' in kwargs.keys():
        model_out[-1] = kwargs['model_in']
    elif 'm0' in kwargs.keys():
        model_out[-1] = quiescence(T, kwargs['m0'])

    dF = 2.512 ** dm - 1.

    rise = (T <= 0) & (T >= -1)
    decay = (T > 0) & (T <= 30)

    model_out[0][rise] = _flare_rise(T[rise], dF=dF)
    model_out[0][decay] = _flare_decay(T[decay], dF=dF)

    model_out[0] = -np.log10(model_out[0] + 1.) / np.log10(2.512)  # Converts back to mags

    return np.sum(model_out, axis=0)


def _N_flare_model(t, m0, **model_pars):
    """
    Computes a composite flare model for given time array, quiescent magnitude,
    and set of flare parameters. 
    
    Used in fitting and plotting functions (pyvan.fit_flares and pyvan.get_flare_model).
    
    Parameters
    ----------
    t : 1-d array_like
        The array of time values for which to evaluate the flare model
        
    m0 : float
        The quiet level of the light-curve in magnitudes
        
    model_pars : dict
        Captures the additional flare parameters, allowing us to fit for any number of flare events at once in LMFit,
        without needing to explicitly define each parameter.
        For each of N flare events to be added to the model (i = 1 to i = N), contains:
        - t0_i : float
            The time corresponding to the beginning of the flare event's rise phase.
            
        - dt_i : float
            The duration of the flare event's rise phase from t0_i.
            
        - dm_i : float
            The magnitude amplitude of the event above m0. Note: this parameter is
            positively defined such that it gives the number of magnitudes that the
            flare event peak is brighter than the quiescence.

    Returns
    -------
    flare_model : 1-d array
        The magnitude array of the composite flare model evaluated at t
    """

    N = len(list(model_pars.keys())[::3])

    model = np.zeros((N + 1, len(t)))
    model[-1] = quiescence(t, m0)

    for i in range(N):
        t0, dt, dm = model_pars['t0_' + str(i + 1)], model_pars['dt_' + str(i + 1)], model_pars['dm_' + str(i + 1)]
        model[i] = _flarefn(t, t0, dt, dm)

    flare_model = np.sum(model, axis=0)

    return flare_model


def fit_flares(data, expt, flare_cands=get_cands, N_concurrent=3):
    """
    Carries out fitting of all flare candidates in a target light-curve.

    Parameters
    ----------
    data : structured ndarray
        Light-curve data containing at least columns labeled 'mjd', 'mag', and 'magErr'.

    expt : float
        The exposure time in seconds for observations in the light-curve. This parameter is only used for informing the
        bounds some of the flare parameters.

    flare_cands : callable or ndarray, optional
        If callable: A function that takes 'data' as its argument, and returns an array of candidate flare peak indices
        for the data. If ndarray: an array of indices corresponding to candidate flare peaks in "data". These indices
        should be arranged in the order that you want them fit; i.e. if you leave "N_concurrent" as 3, the first 3
        indices in this array will be fit at one time, and any subsequent candidates are appended to the resulting model
        thereafter. Generally, you probably want your largest amplitude candidates first. Default option is a function
        identifying candidates as described in Lawson+(2019).

    N_concurrent: int, optional
        The number of flare events to fit simultaneously. A light-curve with many events will take a LONG time to fit if
        this parameter is changed to a large number. Alternatively, if too low, may allow a single event to determine a
        quiescent magnitude that fits poorly to all other events. In practice with PTF data, the default value of 3
        seemed to be the sweet spot. May warrant more consideration if you're fitting data with a large number of
        candidates per light-curve, or with a huge number of observations overall. This could also be used for fitting
        a single complex flare event.

    Returns
    -------
    : dict
        A dictionary containing information regarding the achieved fit.
    """
    if callable(flare_cands):
        peaks = flare_cands(data)
    else:
        peaks = flare_cands
    if len(peaks) == 0:
        return {'fit': False, 'logL': np.nan}
    expt = expt / 86400.  # Converting from seconds to days
    N = (len(peaks) if len(peaks) <= N_concurrent else N_concurrent)

    fit_params = []
    m0_init, m0_err, t0_init, dt_init, dm_init = initialize_flare_params(data, peaks)
    fmodel = lmfit.Model(_N_flare_model)
    fmodel.set_param_hint('m0', vary=True, value=m0_init, min=(m0_init - 10 * m0_err), max=(m0_init + 10 * m0_err))
    for i in range(N):
        fmodel.set_param_hint('t0_' + str(i + 1), vary=True, value=t0_init[i], min=(t0_init[i] - 0.08),
                              max=(t0_init[i] + dt_init[i] - expt))
        fmodel.set_param_hint('dm_' + str(i + 1), vary=True, value=dm_init[i], min=0., max=(dm_init[i] + 5.))
        fmodel.set_param_hint('dt_' + str(i + 1), vary=True, value=dt_init[i], min=expt / 5., max=0.5)

    suppressed_peaks = peaks[N:]
    weights = 1. / data['magErr']
    weights[suppressed_peaks] = 0.
    result = fmodel.fit(data['mag'], t=data['mjd'], weights=weights, method='differential_evolution')
    fit_params.append(result.best_values['m0'])

    for i in range(N):
        fit_params.append(result.best_values['t0_' + str(i + 1)])
        fit_params.append(result.best_values['dt_' + str(i + 1)])
        fit_params.append(result.best_values['dm_' + str(i + 1)])
    fit_params = np.array(fit_params)
    current_model = result.best_fit

    if N < len(peaks):
        for i in range(N, len(peaks)):
            fmodel = lmfit.Model(_flarefn)
            fmodel.set_param_hint('t0', vary=True, value=t0_init[i], min=(t0_init[i] - 0.08),
                                  max=(t0_init[i] + dt_init[i] - expt))
            fmodel.set_param_hint('dm', vary=True, value=dm_init[i], min=0., max=(dm_init[i] + 5.))
            fmodel.set_param_hint('dt', vary=True, value=dt_init[i], min=expt / 5., max=0.5)

            result = fmodel.fit(data['mag'], t=data['mjd'], weights=1. / data['magErr'],
                                method='differential_evolution', model_in=current_model)
            fit_params = np.append(fit_params, result.best_values['t0'])
            fit_params = np.append(fit_params, result.best_values['dt'])
            fit_params = np.append(fit_params, result.best_values['dm'])
            current_model = result.best_fit
    chisq = chi_squared(data['mag'], current_model, data['magErr'])
    return {'chisq': chisq, 'logL': log_likelihood(chisq, len(data)), 'params': fit_params, 'peaks': peaks, 'fit': True}


def get_flare_model(fit_dict, high_res=False):
    """
    Used to build a flare model from the fit dictionary of a flare candidate.

    Parameters
    ----------
    fit_dict : dict
        The fit dictionary for a target, containing at least the flare fit dictionary ['flare'] and the data array
         ['data'].

    high_res : bool, optional
        If True, produces a model in which the times are interpolated to produce a nice looking flare function
        for presentation.

    Returns
    -------
    t : ndarray
        Times in days corresponding to the entries in 'm'

    model: ndarray
        Magnitudes for the flare model
    """
    params = fit_dict['flare']['params']

    if high_res:
        t = finer_time_resolution(params, fit_dict['data'])
    else:
        t = fit_dict['data']['mjd']

    n_flares = int((len(params) - 1) / 3)
    model_pars = {}

    for i in range(n_flares):  # t0, dm, dt
        model_pars['t0_' + str(i + 1)] = params[1:][i * 3]
        model_pars['dt_' + str(i + 1)] = params[1:][i * 3 + 1]
        model_pars['dm_' + str(i + 1)] = params[1:][i * 3 + 2]

    model = _N_flare_model(t, params[0], **model_pars)

    return [t, model]


def initialize_flare_params(data, peaks):
    """
    Initial guesses at flare parameters by which pyvan.fit_flares() determines bounds.
    """
    m0 = np.median(data['mag'])
    m0_err = med_err(data['mag'])
    dt = []
    dm = []
    t0 = []
    for i in range(len(peaks)):
        dm.append(m0 - data['mag'][peaks[i]])
        t0.append(find_apparent_t0(data, peaks[i], m0, dm[i]))
        dt.append(data['mjd'][peaks[i]] - t0[i])
    return m0, m0_err, t0, dt, dm


def analytic_rrlyrae(t, t0, m0, dt, dm):
    """
    Used by fit_rrlyrae to evaluate an RR Lyrae model using a template globally defined in fit_rrlyrae. You probably
    want to use rrlyrae_with_template instead for plotting/inspection.

    Parameters
    ----------
    t : 1-d array_like
        The array of time values for which to evaluate the RR Lyrae model

    t0 : float
        Oscillation period starting point in days. This is somewhat like the phase of the model, just serving to shift
        the oscillations left or right in time

    m0 : float
        "Base" magnitude for the RR Lyrae template. This is just the magnitude level corresponding to the
        dimmest point in the oscillations

    dt : float
        Period of oscillation in days

    dm : float
        Amplitude of oscillations from m0

    Returns
    -------
    : ndarray
        Magnitudes for the RR Lyrae model
    """
    T = (t - t0) % dt / dt
    y = fn(T)
    return dm * (y - 1.) + m0


def rrlyrae_with_template(t, params, template):
    """
    Used for plotting RR Lyrae fits.

    Parameters
    ----------
    t : 1-d array_like
        The array of time values for which to evaluate the RR Lyrae model

    params : array-like
        Set of parameters at which to evaluate the template [t0, m0, dt, dm]

    template: str
        Template ID to use in creating the RR Lyrae model

    Returns
    -------
    : ndarray
        Magnitudes for the RR Lyrae model
    """
    for filt in rrl_template_dict:
        if template in rrl_template_dict[filt].keys():
            fn = rrl_template_dict[filt][template]
    t0, m0, dt, dm = params
    T = (t - t0) % dt / dt
    y = fn(T)
    return dm * (y - 1.) + m0


def rrl_bounds(data):
    """
    Generates bounds for each of the RR Lyrae template parameters. Ranges for period (dt) and amplitude (dm) are static,
    based on literature values, while ranges for the time offset (t0) and base magnitude (m0) are set based on the
    data. Though one might imagine constraining period and amplitude based on evidence in the data, this was found to
    produce more issues than it solved.
    """
    t0_guess = data['mjd'][int(len(data) / 2.)]
    m0_guess = weighted_quantile(data['mag'], np.array([.95]), sample_weight=1. / data['magErr'], old_style=True)

    m0_range = m0_guess + np.array(
        [-0.1, 1.5])  # ie  dimmest mag shouldn't be much brighter than dimmest 5th percentile,
    # but could be much dimmer with sparse sampling
    dm_range = [0.1, 1.5]
    dt_range = [0.08, 1.6]
    t0_range = [(t0_guess - dt_range[1] / 2.), (t0_guess + dt_range[1] / 2.)]
    return [t0_range, m0_range, dt_range, dm_range]


def fit_rrlyrae(data, filt, K=12):
    """
    Fits a light-curve with empirical RR Lyrae templates. See Lawson+(2019) for details regarding the technique here.

    Parameters
    ----------
    data : structured ndarray
        Light-curve array containing columns 'mjd', 'mag', and 'magErr'

    filt: str
        Identifier for the ugriz filt (default) of the data. Used to fetch appropriate templates.

    K : int, optional
        Number of times that the light-curve will be fit with the first template.

    Returns
    -------
    : dict
        Contains:
            'chisq', float - the best-fit template's chi-squared value

            'logL', float - the best-fit template's log-likelihood value

            'params', ndarray -  array containing the best-fit template parameters in the order [t0, m0, dt, dm]

            'template', str - the string identifier of the best-fit RR Lyrae template

            'fit', bool - indicates that the target was fit for RR Lyrae
    """
    template_ids = rrl_template_dict[filt]['ordered_keys']

    n = len(data)
    t0_bounds, m0_bounds, dt_bounds, dm_bounds = rrl_bounds(data)
    init_vals = np.array([np.mean(t0_bounds), np.mean(m0_bounds), np.mean(dt_bounds), np.mean(dm_bounds)])

    logL_list = []
    chisq_list = []
    fit_params = []

    vals = np.copy(init_vals)
    for i in range(len(template_ids)):
        global fn
        fn = rrl_template_dict[filt][template_ids[i]]
        fmodel = lmfit.Model(analytic_rrlyrae)
        fmodel.set_param_hint('t0', vary=True, value=vals[0], min=t0_bounds[0], max=t0_bounds[1])
        fmodel.set_param_hint('m0', vary=True, value=vals[1], min=m0_bounds[0], max=m0_bounds[1])
        fmodel.set_param_hint('dt', vary=True, value=vals[2], min=dt_bounds[0], max=dt_bounds[1])
        fmodel.set_param_hint('dm', vary=True, value=vals[3], min=dm_bounds[0], max=dm_bounds[1])
        if i == 0:
            zeroth_params = []
            zeroth_chisqs = []
            zeroth_logL = []
            for _ in range(K):
                result = fmodel.fit(data['mag'], t=data['mjd'], weights=1. / data['magErr'],
                                    method='differential_evolution')
                zeroth_chisqs.append(result.chisqr)
                zeroth_logL.append(log_likelihood(result.chisqr, n))
                zeroth_params.append(lmfit_params_to_list(result.best_values, ['t0', 'm0', 'dt', 'dm']))
            best_index = np.argmax(zeroth_logL)
            chisq_list.append(zeroth_chisqs[best_index])
            logL_list.append(zeroth_logL[best_index])
            vals = zeroth_params[best_index]
            fit_params.append(vals)

        else:
            result = fmodel.fit(data['mag'], t=data['mjd'], weights=1. / data['magErr'])
            chisq_list.append(result.chisqr)
            logL_list.append(log_likelihood(result.chisqr, len(data)))
            vals = lmfit_params_to_list(result.best_values, ['t0', 'm0', 'dt', 'dm'])
            fit_params.append(vals)
    best_index = np.argmax(np.array(logL_list))
    return {'chisq': chisq_list[best_index], 'logL': log_likelihood(chisq_list[best_index], len(data)),
            'params': fit_params[best_index], 'template': template_ids[best_index], 'fit': True}


def fit_quiescence(data):
    """
    Fits a light-curve with a flat line for gauging the statistical significance of any outliers

    Parameters
    ----------
    data : structured ndarray
        Light-curve array containing columns 'mjd', 'mag', and 'magErr'

    Returns
    -------
    : dict
        Contains:
            'chisq', float - the best-fit template's chi-squared value

            'logL', float - the best-fit template's log-likelihood value

            'param', ndarray -  array containing the best-fit template parameter (m0)

            'fit', bool - indicates that the target was fit for quiescent profile

    """
    n = len(data)
    popt, _pcov = curve_fit(quiescence, data['mjd'], data['mag'], sigma=data['magErr'])
    chisq = chi_squared(data['mag'], quiescence(data['mjd'], popt[0]), data['magErr'])
    return {'chisq': chisq, 'logL': log_likelihood(chisq, n), 'param': popt[0], 'fit': True}


def lmfit_params_to_list(best_params, param_names):
    """
    Gets a list of best-fit parameters values out of the lmfit result object
    """
    par_list = []
    for param in param_names:
        par_list.append(best_params[param])
    return par_list


def chi_squared(y_obs, y_model, error):
    """
    Computes the chi-squared value for observations "y_obs" with uncertainties "error" fit with model "y_model".
    """
    residuals = y_obs - y_model
    return np.sum((residuals / error) ** 2)


def log_likelihood(chisq, n):
    """
    Computes the log-likelihood of a fit from the chi-squared value and the number of observations.
    """
    return (-n / 2.) * np.log(chisq / n)


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """ Vectorized computation of weighted quantiles, courtesy of user Alleo on stackoverflow:
    NOTE: quantiles should be in [0, 1]
    :param values: np.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with np.percentile.
    :return: np.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def med_err(arr):
    """
    Computes the standard error on the median for an array of values.
    """
    return np.std(arr) / np.sqrt(len(arr))


def make_template_dict():
    """
    Creates a dictionary of interpolated RR Lyrae templates from .dat files in pyvan/rrlyr_templates. Users could
    supply additional templates for other filters to the directory as well, with names like: '*F.dat' where 'F' is a
    single letter/number identifier for the filter, and where the numbers should roughly follow the steepness of
    the template. Alternatively, a single template for a filter should work alright as well. Template '.dat' files
    should contain a single period for the RR Lyrae in the form of two tab delimited columns, the first of which is
    the phase ranging from 0 to 1, and the second is the amplitude, ranging from 0 to 1.

    Parameters
    ----------

    Returns
    -------
    fn_dict : dict
        Default, contains keys for each of the ugriz filters: 'u', 'g', 'r', 'i', 'z'. Each of these contains numerical
        keys corresponding to the array of RR Lyrae templates from the source
    """
    data_path = pkg_resources.resource_filename('pyvan', 'rrlyr_templates/')
    all_template_files = glob.glob(data_path + '*.dat')
    filters = np.unique([x.split('/')[-1].split('.')[0][-1] for x in all_template_files])
    fn_dict = {}
    for filt in filters:
        fn_dict[filt] = {}
        template_files = glob.glob(data_path + '*' + filt + '.dat')
        template_ids, t_idx = sort_template_ids([x.split('/')[-1].split('.')[0] for x in template_files],
                                                return_index=True)
        template_files = np.asarray(template_files)[t_idx]
        fn_dict[filt]['ordered_keys'] = template_ids
        for i in range(len(template_ids)):
            t, y = np.loadtxt(template_files[i]).T
            tlong = np.concatenate([t, [t[0] + 1.]])
            ylong = np.concatenate([y, [y[0]]])
            fn_dict[filt][template_ids[i]] = interp1d(tlong, ylong, kind='cubic')
    return fn_dict


def sort_template_ids(template_ids, return_index=True):
    """
    Sorts template IDs to flow well in terms of steepness --- helps pyvan.fit_rrlyrae() converge well, and improves
    efficiency quite a bit (over just using differential evolution to fit each template individually).
    """
    template_nums = np.array([int(tid[:-1]) for tid in template_ids])
    tn_idx = np.argsort(template_nums)[::-1]
    if return_index:
        return np.asarray(template_ids)[tn_idx], tn_idx
    return np.asarray(template_ids)[tn_idx]


rrl_template_dict = make_template_dict()


def save_buttons(buttons):
    global button_save
    button_save = buttons


def plot_all_fits(tar_fit, high_res=True, templates=None, x_range=None, y_range=None, donor_lightcurve=None):
    """
    Takes the fit dictionary for a target and displays a light-curve along with widget buttons to overlay any desired
    model fits. Best used in a Jupyter Notebook (I have not tested this in other environments!). Make sure to enable
    matplotlib interactive mode in your notebook before you begin, e.g.:

        import matplotlib.pyplot as plt
        %matplotlib notebook

        pyvan.plot_all_fits(my_target_tar_fit)

    Also displays the log-likelihood value for each template on its respective button.

    Parameters
    ----------
    tar_fit : dict
        A dictionary of PyVAN fit information for a single target. i.e. the product of pyvan.fit_target

    high_res : bool, optional
        If True, will evaluate the best-fit models at very fine time resolution to produce smooth models. If False,
        evaluates models at the observed times only.

    templates : list, optional
        A list of templates for which to display models. Elements may be: strings for default templates ('flare',
        'quiescent', and 'rrlyrae') or for fits computed using the pyvan.fit_general function. In the latter case, the
        string should match the key for the template's fit in "tar_fit". You may also use this option if you've written
        your own template fitting procedure, so long as the resulting dictionary for the template includes the key 'fn',
        corresponding to the template function which takes arguments as: "fn(times, *params)" and returns the model
        evaluated at those times and with those parameters. The dictionary also needs to have key 'logL' corresponding
        to the float log-likelihood for the best-fit template, and 'fit', a boolean indicating that the template fit was
        successfully completed. Alternatively, each entry can be a dictionary (like the ones passed into pyvan.fit or
        pyvan.fit_target for fitting via the pyvan.fit_general procedure). This allows including the additional key
        'plot_sampling'- callable, array-like or integer. If callable: a function which takes "fit dict" as its
        argument, and returns an array of time values at which to evaluate the template. If array-like: evaluates the
        template at these times. If integer: generates this many evenly spaced times between the smallest and largest
        times in the 'data' array. Note: to avoid more clever solutions for how to space buttons, this function will
        only display 7 templates at a time. If passed more than 7 entries in "templates", the list is truncated to only
        the display the first 7. Default is 'None', which simply plots the default templates.

    x_range : tuple, optional
        Tuple containing minimum and maximum x-axis values as (xmin,xmax) for framing the plot. By default, the extent
        of the best-fitting template determines this boundary

    y_range : tuple, optional
        Tuple containing minimum and maximum y-axis values as (ymin,ymax) for framing the plot. By default, the extent
        of the best-fitting template determines this boundary. Note: if you want an inverted magnitude axis, simply
        invert your tuple.

    donor_lightcurve : structured ndarray, optional
        If you're plotting fits to a "simulated" light-curve (as utilized in Lawson+(2019)), you can pass in the
        original "donor" light-curve as well to see it plotted alongside your fits and simulated data. Defaults to None.

    Returns
    -------

    """
    if isinstance(templates, NoneType):
        templates = np.array(['flare', 'quiescent', 'rrlyrae'])
    border_width = 1.5
    data = tar_fit['data']
    if len(templates) > 7:
        Nbuttons = 8
        templates = templates[:7]
    else:
        Nbuttons = len(templates) + 1
    label_list = ['Clear']
    t_list = [[]]
    y_list = [[]]
    logL_arr = np.array([-np.inf])
    for template in templates:
        t_template, y_template = [], []
        if isinstance(template, dict):
            template_label = template['name']
            logL_arr = np.append(logL_arr, tar_fit[template_label]['logL'])
            if tar_fit[template_label]['fit']:
                if 'plot_sampling' in template:
                    if callable(template['plot_sampling']):
                        t_template = template['plot_sampling'](tar_fit)
                    elif type(template['plot_sampling']) == int:
                        t_template = np.linspace(np.min(data['mjd']), np.max(data['mjd']), template['plot_sampling'])
                    elif type(template['plot_sampling']) == np.ndarray or type(template['plot_sampling']) == list:
                        t_template = np.asarray(template['plot_sampling'])
                    else:
                        raise TypeError(
                            'Template dictionary key "plot_sampling" (if specified) must either refer to: a callable, '
                            'an integer, or an array-like of times')
                elif high_res:
                    t_template = np.linspace(np.min(data['mjd']), np.max(data['mjd']), 500000)
                else:
                    t_template = data['mjd']
                y_template = template['fn'](t_template, *tar_fit[template['name']]['params'])
        elif template == 'flare':
            template_label = 'Flare'
            logL_arr = np.append(logL_arr, tar_fit[template]['logL'])
            if tar_fit[template]['fit']:
                t_template, y_template = get_flare_model(tar_fit, high_res=high_res)
        elif template == 'quiescent':
            template_label = 'Quiescent'
            logL_arr = np.append(logL_arr, tar_fit[template]['logL'])
            if tar_fit[template]['fit']:
                t_template = np.array([data['mjd'].min(), data['mjd'].max()])
                y_template = quiescence(t_template, tar_fit[template]['param'])
        elif template == 'rrlyrae':
            template_label = 'RR Lyrae'
            logL_arr = np.append(logL_arr, tar_fit[template]['logL'])
            if tar_fit[template]['fit']:
                if high_res:
                    t_template = np.linspace(data['mjd'].min(), data['mjd'].max(), 500000)
                else:
                    t_template = data['mjd']
                y_template = rrlyrae_with_template(t_template, tar_fit[template]['params'],
                                                   tar_fit[template]['template'])
        elif type(template) == str:
            if template not in tar_fit or 'fn' not in tar_fit[template]:
                raise ValueError(
                    'If an item in "templates" is a string, it must either be a default template ("flare", '
                    '"quiescent", or "rrlyrae"), or must be a key for "tar_fit" corresponding to a dictionary'
                    'containing the template function for key "fn"')
            template_label = template
            logL_arr = np.append(logL_arr, tar_fit[template]['logL'])
            if tar_fit[template]['fit']:
                if high_res:
                    t_template = np.linspace(data['mjd'].min(), data['mjd'].max(), 500000)
                else:
                    t_template = data['mjd']
                y_template = tar_fit[template]['fn'](t_template, *tar_fit[template]['params'])
        else:
            raise TypeError(
                'Entries of argument "templates" must either be: a string corresponding to a fit dictionary '
                'contained within "tar_fit", or a dictionary. See documentation for details.')
        t_list.append(t_template)
        y_list.append(y_template)
        label_list.append(template_label)

    fleft, fbottom, ftop, fright = 0.15, 0.15, 0.85, 0.785
    axis_color = 'ghostwhite'
    fig = plt.figure(figsize=(18, 5))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=fleft, bottom=fbottom, top=ftop, right=fright)

    if isinstance(donor_lightcurve, NoneType):
        ncol = 2
        ax.scatter(data['mjd'], data['mag'], s=80, c='dimgray', edgecolor='black', linewidth=2., zorder=3,
                   label='Target Data')
        ax.errorbar(data['mjd'], data['mag'], yerr=data['magErr'], linestyle="None", c='black', zorder=2, capthick=2,
                    capsize=10, elinewidth=2)

    elif isinstance(donor_lightcurve, np.ndarray):
        ncol = 3
        ax.scatter(donor_lightcurve['mjd'], donor_lightcurve['mag'], c='black', alpha=0.75, s=4, zorder=0)
        ax.scatter([], [], c='k', alpha=0.75, s=35,
                   label='Original Donor Data')  # Dummy plot to produce a more visible point for the key
        ax.scatter(data['mjd'], data['mag'], s=100, c='lightgrey', edgecolor='black', linewidth=2, zorder=3,
                   label='Simulated Data')
        ax.errorbar(data['mjd'], data['mag'], yerr=data['magErr'], linestyle="None", c='black', zorder=2, capthick=2,
                    capsize=10, elinewidth=2)

    else:
        raise TypeError('If "donor_lightcurve" is specified, it must be None or a numpy structured array having columns'
                        ' "mjd", "mag", and "magErr")')

    best = np.argmax(logL_arr)
    [line] = ax.plot(t_list[best], y_list[best], c='orange', lw=2., label='Fit to Data', zorder=1, alpha=0.8)

    def _button_0_on_clicked(mouse_event):
        line.set_xdata([])
        line.set_ydata([])
        fig.canvas.draw_idle()

    def _button_1_on_clicked(mouse_event):
        line.set_xdata(t_list[1])
        line.set_ydata(y_list[1])
        fig.canvas.draw_idle()

    def _button_2_on_clicked(mouse_event):
        line.set_xdata(t_list[2])
        line.set_ydata(y_list[2])
        fig.canvas.draw_idle()

    def _button_3_on_clicked(mouse_event):
        line.set_xdata(t_list[3])
        line.set_ydata(y_list[3])
        fig.canvas.draw_idle()

    def _button_4_on_clicked(mouse_event):
        line.set_xdata(t_list[4])
        line.set_ydata(y_list[4])
        fig.canvas.draw_idle()

    def _button_5_on_clicked(mouse_event):
        line.set_xdata(t_list[5])
        line.set_ydata(y_list[5])
        fig.canvas.draw_idle()

    def _button_6_on_clicked(mouse_event):
        line.set_xdata(t_list[6])
        line.set_ydata(y_list[6])
        fig.canvas.draw_idle()

    def _button_7_on_clicked(mouse_event):
        line.set_xdata(t_list[7])
        line.set_ydata(y_list[7])
        fig.canvas.draw_idle()

    def _button_8_on_clicked(mouse_event):
        line.set_xdata(t_list[8])
        line.set_ydata(y_list[8])
        fig.canvas.draw_idle()

    button_fns = [_button_0_on_clicked, _button_1_on_clicked, _button_2_on_clicked, _button_3_on_clicked,
                  _button_4_on_clicked, _button_5_on_clicked, _button_6_on_clicked, _button_7_on_clicked,
                  _button_8_on_clicked]

    b_left, b_width, b_height = 0.8, 0.1, 0.075
    b_gap = 0.015  # Space between buttons

    mpt = fbottom + (ftop - fbottom) / 2.  # midpoint between ftop and fbottom
    b_space = Nbuttons * b_height + (Nbuttons - 1) * b_gap

    buttons = []
    for i in np.arange(Nbuttons):
        label = label_list[i]
        if i > 0:
            label += ' ($\ell =$' + str(np.round(logL_arr[i], 1)) + ')'
        b_bottom = mpt - b_space / 2. + i * (b_height + b_gap)
        button_ax = fig.add_axes([b_left, b_bottom, b_width, b_height])
        [border.set_linewidth(border_width) for border in button_ax.spines.itervalues()]
        button_i = Button(button_ax, label, color=axis_color, hovercolor='0.925')
        button_i.on_clicked(button_fns[i])
        buttons.append(button_i)

    ax.set_xlabel("Time (days)", fontsize=16)
    ax.set_ylabel("Magnitude", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_ylim(ax.set_ylim()[::-1])

    ax.tick_params(axis='both', which='major', labelsize=14, right=True, top=True, direction='in', length=10,
                   width=border_width, pad=5, zorder=0)
    [border.set_linewidth(border_width) for border in ax.spines.itervalues()]

    lgnd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=ncol, framealpha=1.,
                     prop={'family': 'Serif', 'size': 12}, edgecolor='black', facecolor=axis_color, fancybox=False)
    lgnd.get_frame().set_linewidth(border_width)

    if not isinstance(x_range, NoneType):
        ax.set_xlim(x_range[0], x_range[1])

    if not isinstance(y_range, NoneType):
        ax.set_ylim(y_range[0], y_range[1])

    plt.show()
    save_buttons(buttons)


def logL_kde_scatter(fit_dict, x_param='flare-quiescent', y_param='flare-rrlyrae', marker_size=80, x_range=None,
                     y_range=None, kde_cmap='Greens', xy_cut=None):
    """
    Takes a dictionary of fit targets, and generates a kernel density estimate colored scatter plot of two relative
    log-likelihood metrics. Returns the figure and figure axis so users can append or alter anything not allowed
    explicitly by argument options. Best used in a Jupyter Notebook, but may work fine via command line execution as
    well.

    Parameters
    ----------
    fit_dict : dict
        dictionary of fits to targets resulting from pyvan.fit()

    x_param : string, optional
        string corresponding to a key in fit_dict[i]['rel_fit'] which will be used to fetch the array of x axis values
        for plotting. By default, plots flare-quiescent along the x-axis.

    y_param : string, optional
        string corresponding to a key in fit_dict[i]['rel_fit'] which will be used to fetch the array of y axis values
        for plotting. By default, plots flare-rrlyrae along the y-axis.

    marker_size : int or float, optional
        size for markers in the plot. For a KDE scatter plot you'll generally want larger markers than you might
        normally use for Python scatter plots, since the issue of overlap is handled coloring points based on density.
        Default value is 80.

    x_range : tuple, optional
        tuple as (xmin, xmax), the x-axis limits for the plot. Default is None.

    y_range : tuple, optional
        tuple as (ymin, ymax), the y-axis limits for the plot. Default is None.

    kde_cmap : string
        Name of matplotlib colormap to color the density axis in the plot. Default is 'Greens'.

    xy_cut : tuple
        Draws dashed lines at these values to delineate regions of the difference of log-likelihood space that will
        survive cuts to eliminate contaminants. For the study conducted in Lawson+(2019), we found that flare-quiescent
        and flare-rrlyrae cuts of 10.44 and 11.26 respectively produced a flare sample containing <1% of any contaminant
        population tested. I could only conjecture as to how this might change (if at all) for groups utilizing data
        with quality much better or worse than the PTF-quality data we utilized.

    Returns
    -------
    fig : the matplotlib figure object

    ax : matplotlib axis object onto which the scatter plot has been drawn.
    """
    border_width = 1.5
    fig = plt.figure(figsize=(10.75, 10))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.875)

    x_arr, y_arr = np.array([]), np.array([])
    for tar in fit_dict:
        x_arr = np.append(x_arr, fit_dict[tar]['rel_fit'][x_param])
        y_arr = np.append(y_arr, fit_dict[tar]['rel_fit'][y_param])

    xy = np.vstack([x_arr, y_arr])
    z = gaussian_kde(xy)(xy)  # Density values for each (x,y) pair

    idx = z.argsort()  # Indices producing arrays sorted by z value of each (x,y,z)
    x, y, z = x_arr[idx], y_arr[idx], z[idx] / z.max()  # Sorting arrays by previous line

    ax.scatter(x, y, s=marker_size, c=z, cmap=kde_cmap, edgecolor='', norm=mpl.colors.Normalize(vmin=-0.2, vmax=1.))

    ax.axhline(0., c='gray', alpha=0.5, zorder=3)
    ax.axvline(0., c='gray', alpha=0.5, zorder=3)

    if not isinstance(xy_cut, NoneType):
        xcut, ycut = xy_cut
        y_max = y_arr.max() if y_arr.max() >= ycut + 10 else ycut + 10
        x_max = x_arr.max() if x_arr.max() >= xcut + 10 else xcut + 10

        ax.plot(np.repeat(xcut, 2), np.array([ycut, y_max]), c='black', ls='dashed', zorder=3)
        ax.plot(np.array([xcut, x_max]), np.repeat(ycut, 2), c='black', ls='dashed', zorder=3)

    ax.tick_params(axis='both', which='major', right=True, top=True, labelsize=15, direction='in',
                   width=border_width, length=8.5)

    if not isinstance(x_range, NoneType):
        ax.set_xlim(x_range[0], x_range[1])

    if not isinstance(y_range, NoneType):
        ax.set_ylim(y_range[0], y_range[1])

    [border.set_linewidth(border_width) for border in ax.spines.itervalues()]

    sct = ax.scatter([], [], c=[], cmap=kde_cmap, edgecolor='', norm=mpl.colors.Normalize(vmin=0.0, vmax=1.))
    cb_ax = fig.add_axes([0.885, 0.11, 0.026, 0.77])
    cbar = fig.colorbar(sct, cax=cb_ax)
    cbar.set_label('Relative Density', fontsize=20)

    cbar.set_ticks([0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=16, direction='in', length=10, width=border_width)
    cbar.outline.set_linewidth(border_width)
    cbar.set_clim(0.0, 1.0)

    xlab1, xlab2 = x_param.split('-')
    ylab1, ylab2 = y_param.split('-')
    ax.set_xlabel('$\ell_{' + xlab1 + '} - \ell_{' + xlab2 + '}$', fontsize=26)
    ax.set_ylabel('$\ell_{' + ylab1 + '} - \ell_{' + ylab2 + '}$', fontsize=26)

    return fig, ax