import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    return np, plt, stats


@app.cell
def _():
    # group sequential design assessment imports
    from py_group_sequential_designs import generate_boundaries as bd
    from py_group_sequential_designs import feasibility_penalty as fp
    from py_group_sequential_designs import boundary_manipulations as fmt_bd
    from py_group_sequential_designs import function_to_minimize as fn_min
    from py_group_sequential_designs import generate_gpr_input as gen_input
    from py_group_sequential_designs import simulate as sim
    from py_group_sequential_designs import sample_size as ss

    return bd, fmt_bd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Boundary intuition
    """)
    return


@app.cell
def _(np, stats):
    _bounds = [1.99218512e+00, 1.99218512e+00, 1.99218512e+00, -1.99218512e+00, -1.99218512e+00]
    lower = np.empty(6)
    upper = np.empty(6)

    for _i, _bound in enumerate(_bounds):
        lower[_i] = stats.norm.ppf(q=[0.3,0.7], loc=_bound, scale=0.5)[0]
        upper[_i] = stats.norm.ppf(q=[0.3,0.7], loc=_bound, scale=0.5)[1]

    # add a single nan so it works with the plotting loops below
    lower[5] = np.nan
    upper[5] = np.nan

    print(lower)
    print(upper)
    return


@app.cell
def _(np, plt, stats):
    _fig, _axes = plt.subplots(nrows=3, ncols=3, figsize=(12,8))
    xx = np.linspace(start=-4, stop=4, num=200)

    pocock_bounds = [ 1.99218512e+00, 1.99218512e+00, 1.99218512e+00, -1.99218512e+00, -1.99218512e+00, 1.99218512e+00]
    of_bounds = [ 2.96112971e+00, 2.09383490e+00, 1.70960903e+00, -2.96112971e+00, -2.09383490e+00, 1.70960903e+00]
    tri_bounds = [ 2.11957748e+00, 1.87345951e+00, 1.83560794e+00, 6.28553399e-16, 1.12407571e+00, 1.83560794e+00]
    row_labels = ["Analysis 1", "Analysis 2", "Analysis 3"]
    col_labels = ["Pocock", "O'Brien-Fleming", "Triangular"]

    for _i in range(3):
        for _j in range(3):
            _axes[_i][_j].plot(xx, stats.norm.pdf(x=xx),
                               label="Null")

    for _i in range(3):
        for _j in range(3):
            _axes[_i][_j].plot(xx, stats.norm.pdf(x=xx, loc=1), 
                               color = "orange",
                               label="Alternative")

    # plot the pocock bounds
    for _i in range(3):
        _axes[_i,0].axvline(x=pocock_bounds[_i], color = "green")
        _axes[_i,0].axvline(x=pocock_bounds[_i+3], color = "green")

    # plot the obrien-fleming bounds
    for _i in range(3):
        _axes[_i,1].axvline(x=of_bounds[_i], color = "green")
        _axes[_i,1].axvline(x=of_bounds[_i+3], color = "green")

    # plot the triangular bounds
    for _i in range(3):
        _axes[_i,2].axvline(x=tri_bounds[_i], color = "green")
        _axes[_i,2].axvline(x=tri_bounds[_i+3], color = "green")

    # plot possible boundaries
    #for _j in range(3):
    #    for _k in range(3):
    #        _axes[_k,_j].axvline(x=lower[_k], color = "orange")
    #        _axes[_k,_j].axvline(x=upper[_k], color = "orange")
    #        _axes[_k,_j].axvline(x=lower[_k+3], color = "purple")
    #        _axes[_k,_j].axvline(x=upper[_k+3], color = "purple")

    # plot the titles
    for _ax, _col in zip(_axes[0], col_labels):
        _ax.set_title(_col)
    for _ax, _row in zip(_axes[:,0], row_labels):
        _ax.set_ylabel(_row)


    handles, labels = _axes[0][0].get_legend_handles_labels()
    _fig.legend(handles, labels, bbox_to_anchor=(1.13, 0.5))

    _fig.tight_layout()
    _fig
    return of_bounds, pocock_bounds, tri_bounds, xx


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Distribution of boundary values
    """)
    return


@app.cell
def _(bd):
    upper_bounds = []
    lower_bounds = []
    for _i in range(20):
        p = bd.calculate_pocock_boundaries(n_analyses=_i+2)
        b = bd.calculate_of_boundaries(n_analyses=_i+2)
        t = bd.calculate_triangular_boundaries(n_analyses=_i+2)

        upper_bounds.append(p[0])
        upper_bounds.append(b[0])
        upper_bounds.append(t[0])

        lower_bounds.append(p[1])
        lower_bounds.append(b[1])
        lower_bounds.append(t[1])
    return lower_bounds, upper_bounds


@app.cell
def _(lower_bounds, plt, upper_bounds):
    plt.plot([1,2,3], lower_bounds[5])
    plt.plot([1,2,3], upper_bounds[5])
    return


@app.cell
def _(np, upper_bounds):
    upper_bounds_collapsed = np.concatenate(upper_bounds)
    return (upper_bounds_collapsed,)


@app.cell
def _(lower_bounds, np):
    lower_bounds_collapsed = np.concatenate(lower_bounds)
    return (lower_bounds_collapsed,)


@app.cell
def _(plt, upper_bounds_collapsed):
    plt.hist(upper_bounds_collapsed, bins=50)
    return


@app.cell
def _(stats, upper_bounds_collapsed):
    fit_lognorm = stats.fit(data=upper_bounds_collapsed, dist=stats.lognorm, bounds=[(0,10), (0, 10), (0,10)])
    return (fit_lognorm,)


@app.cell
def _(fit_lognorm):
    fit_lognorm
    return


@app.cell
def _(fit_lognorm):
    fit_lognorm.plot()
    return


@app.cell
def _(lower_bounds_collapsed, plt):
    plt.hist(lower_bounds_collapsed, bins=50)
    return


@app.cell
def _(lower_bounds_collapsed, plt):
    plt.hist(lower_bounds_collapsed[lower_bounds_collapsed>-1.8], bins=50)
    return


@app.cell
def _(lower_bounds_collapsed, stats):
    fit_half_lower = stats.fit(
        data=lower_bounds_collapsed[lower_bounds_collapsed>-1.8], 
        dist=stats.weibull_max, 
        bounds=[(0,10), (0, 10), (0,10)]
    )
    return (fit_half_lower,)


@app.cell
def _(fit_half_lower):
    fit_half_lower
    return


@app.cell
def _(fit_half_lower):
    fit_half_lower.plot()
    return


@app.cell
def _(lower_bounds_collapsed, stats):
    fit_other_lower = stats.fit(
        data=-lower_bounds_collapsed[lower_bounds_collapsed<=-1.8], 
        dist=stats.lognorm, 
        bounds=[(0,10), (0,10), (0,10)]
    )
    return (fit_other_lower,)


@app.cell
def _(fit_other_lower):
    fit_other_lower
    return


@app.cell
def _(fit_other_lower):
    fit_other_lower.plot()
    return


@app.cell
def _(stats):
    d1 = stats.lognorm(s=0.7034079900167564, loc=1.6499732459827996, scale=0.6620695872701784)
    d2 = stats.weibull_max(c=1.4223189831377416, loc=2.4206529325505444, scale=1.6074642047246706)
    d3 = stats.lognorm(s=0.8489881767032773, loc=1.8152781479347766, scale=0.5959923139028731)
    return d1, d2, d3


@app.cell
def _(d1, d2, d3, np):
    n = 10000

    weights = [0.5, 0.2, 0.3]
    choices = np.random.choice([0, 1, 2], size=n, p=weights)

    samples = np.select(
        [choices == 0, choices == 1, choices == 2],
        [d1.rvs(size=n), d2.rvs(size=n), d3.rvs(size=n)*-1]
    )
    return (samples,)


@app.cell
def _(lower_bounds_collapsed, np, plt, samples, stats, upper_bounds_collapsed):
    all_bounds = np.concatenate((lower_bounds_collapsed, upper_bounds_collapsed))

    _xx = np.linspace(all_bounds.min(), all_bounds.max(), 500)

    kde = stats.gaussian_kde(samples, bw_method=0.1)

    plt.hist(all_bounds, bins=75, density=True)
    plt.plot(_xx, kde(_xx))
    return (all_bounds,)


@app.cell
def _(lower_bounds_collapsed, np, plt, upper_bounds_collapsed):
    plt.hist(np.concatenate((lower_bounds_collapsed, upper_bounds_collapsed)), bins=75, density=True)
    return


@app.cell
def _(np, samples):
    np.sum(samples>7)/samples.shape
    return


@app.cell
def _(all_bounds, np):
    np.sum(all_bounds>7)/all_bounds.shape
    return


@app.cell
def _(np, samples):
    np.sum(samples<-7)/samples.shape
    return


@app.cell
def _(all_bounds, np):
    np.sum(all_bounds<-7)/all_bounds.shape
    return


@app.cell
def _(np, samples):
    np.sum((samples > 1.7) & (samples < 4))/samples.shape
    return


@app.cell
def _(all_bounds, np):
    np.sum((all_bounds > 1.7) & (all_bounds < 4))/all_bounds.shape
    return


@app.cell
def _(np, samples):
    np.sum((samples > -5) & (samples < -2.2))/samples.shape
    return


@app.cell
def _(all_bounds, np):
    np.sum((all_bounds > -5) & (all_bounds < -2.2))/all_bounds.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Distribution of boundary differences
    """)
    return


@app.cell
def _(np, upper_bounds):
    upper_bounds_diffs = []
    for bound in upper_bounds:
        upper_bounds_diffs.append(np.diff(bound))
    return (upper_bounds_diffs,)


@app.cell
def _(np, upper_bounds_diffs):
    upper_diffs = np.concatenate(upper_bounds_diffs)
    return (upper_diffs,)


@app.cell
def _(lower_bounds, np):
    lower_bounds_diffs = []
    for _bound in lower_bounds:
        lower_bounds_diffs.append(np.diff(_bound))
    return (lower_bounds_diffs,)


@app.cell
def _(lower_bounds_diffs, np):
    lower_diffs = np.concatenate(lower_bounds_diffs)
    return (lower_diffs,)


@app.cell
def _(plt, upper_diffs):
    plt.hist(upper_diffs, density=True, bins=50)
    return


@app.cell
def _(lower_diffs, plt):
    plt.hist(lower_diffs, density=True, bins=50)
    return


@app.cell
def _(lower_diffs, stats):
    fit_lower_diffs = stats.fit(
        data=lower_diffs[lower_diffs>0], 
        dist=stats.expon, 
        bounds=[(-10,10), (-10, 10)]
    )
    return (fit_lower_diffs,)


@app.cell
def _(fit_lower_diffs):
    fit_lower_diffs
    return


@app.cell
def _(fit_lower_diffs):
    fit_lower_diffs.plot()
    return


@app.cell
def _(stats, upper_diffs):
    fit_upper_diffs = stats.fit(
        data=-(upper_diffs[upper_diffs<0]), 
        dist=stats.expon, 
        bounds=[(-10,10), (-10, 10)]
    )
    return (fit_upper_diffs,)


@app.cell
def _(fit_upper_diffs):
    fit_upper_diffs
    return


@app.cell
def _(fit_upper_diffs, plt, stats, xx):
    fit_upper_diffs.plot()
    plt.plot(
        xx[xx>0],
        stats.expon.pdf(x=xx[xx>0], loc=-4.391041246609717e-05, scale=0.33)
    )
    return


@app.cell
def _(stats):
    1-stats.expon.cdf(x=1, loc=-4.391041246609717e-05, scale=0.33)
    return


@app.cell
def _(np, upper_diffs):
    np.mean(-upper_diffs > 1)
    return


@app.cell
def _(upper_bounds):
    len(upper_bounds)
    return


@app.cell
def _(np):
    np.array([2,5,8,11,14])-2
    return


@app.cell
def _(np, upper_bounds):
    _n=np.arange(start=1, stop=21, dtype=int)
    index = 3*_n-1
    tri_upper_bounds = [upper_bounds[i] for i in index]
    return (tri_upper_bounds,)


@app.cell
def _(np, tri_upper_bounds):
    tri_upper_diffs = np.diff(np.concatenate(tri_upper_bounds))
    return (tri_upper_diffs,)


@app.cell
def _(plt, tri_upper_diffs):
    plt.hist(tri_upper_diffs, bins=50, density=True)
    return


@app.cell
def _(plt, tri_upper_diffs):
    plt.hist(-(tri_upper_diffs[tri_upper_diffs<0]), bins=50)
    return


@app.cell
def _(stats, tri_upper_diffs):
    fit_tri_upper_diffs = stats.fit(
        data=-(tri_upper_diffs[tri_upper_diffs<0]), 
        dist=stats.expon, 
        bounds=[(-10,10), (-10, 10)]
    )
    return (fit_tri_upper_diffs,)


@app.cell
def _(fit_tri_upper_diffs):
    fit_tri_upper_diffs
    return


@app.cell
def _(fit_tri_upper_diffs):
    fit_tri_upper_diffs.plot()
    return


@app.cell
def _(plt, tri_upper_diffs):
    plt.hist(tri_upper_diffs[tri_upper_diffs<0], bins=200)
    return


@app.cell
def _(tri_upper_diffs):
    tri_upper_diffs[(tri_upper_diffs<0) & (tri_upper_diffs>-0.01)]
    return


@app.cell
def _(tri_upper_diffs):
    tri_upper_diffs[tri_upper_diffs==0]
    return


@app.cell
def _(np, upper_bounds_collapsed):
    print(np.max(upper_bounds_collapsed), np.min(upper_bounds_collapsed))
    return


@app.cell
def _(lower_bounds_collapsed, np):
    print(np.min(lower_bounds_collapsed), np.max(lower_bounds_collapsed))
    return


@app.cell
def _(lower_bounds):
    lower_bounds_no_last = [_bound[:-1] for _bound in lower_bounds]
    return (lower_bounds_no_last,)


@app.cell
def _(lower_bounds_no_last, np):
    lower_bounds_no_last_collapsed = np.concatenate(lower_bounds_no_last)
    return (lower_bounds_no_last_collapsed,)


@app.cell
def _(lower_bounds_no_last_collapsed, np):
    print(np.min(lower_bounds_no_last_collapsed), np.max(lower_bounds_no_last_collapsed))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Distribution of first boundary value
    """)
    return


@app.cell
def _(upper_bounds):
    upper_bounds_first = [_bound[0] for _bound in upper_bounds]
    return (upper_bounds_first,)


@app.cell
def _(plt, upper_bounds_first):
    plt.hist(upper_bounds_first, bins=25)
    return


@app.cell
def _(stats, upper_bounds_first):
    fit_upper_first = stats.fit(
        data=upper_bounds_first, 
        dist=stats.lognorm, 
        bounds=[(-10,10), (-10, 10), (-10,10)]
    )
    return (fit_upper_first,)


@app.cell
def _(fit_upper_first):
    fit_upper_first
    return


@app.cell
def _(fit_upper_first):
    fit_upper_first.plot()
    return


@app.cell
def _(lower_bounds):
    lower_bounds_first = [_bound[0] for _bound in lower_bounds]
    return (lower_bounds_first,)


@app.cell
def _(lower_bounds_first, plt):
    plt.hist(lower_bounds_first, bins=25)
    return


@app.cell
def _(lower_bounds_first, np, stats):
    fit_lower_first = stats.fit(
        data=np.concatenate((np.zeros(10), np.ones(10), (0.5*np.ones(10)), -np.array(lower_bounds_first))), 
        dist=stats.truncnorm, 
        bounds=[(-10,10), (-10, 10), (-10, 10), (-10, 10)]
    )
    return (fit_lower_first,)


@app.cell
def _(fit_lower_first):
    fit_lower_first
    return


@app.cell
def _(fit_lower_first):
    fit_lower_first.plot()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Boundary generation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pseudocode
    1. Select the number of analyses planned.
    2. Using the distribution of the upper and lower bounds, select the first upper and lower bounds.
    3. Using the distribution of differences, select the next bound.
    4. Repeat until reached the number of analyses planned.

    First boundary distributions:
    - Lower bound: Generalized normal(beta=0.38796799701527895, loc=-2.3722985797675546, scale=0.08407511315944749)
    - Upper bound: Log normal(s=1.1186072164739178, loc=1.7916736596538718, scale=1.310525778679183)

    Difference distributions:
    - Lower bound: Exponential(loc=0.04883409413328099, scale=0.692533893064704)
    - Upper bound: Exponential(loc=0.0005023428881223424, scale=0.27006672959109945)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simple exponential difference generator
    """)
    return


@app.cell
def _(fmt_bd, np, stats):
    def boundary_generator(n_analyses):
        lower_bounds = np.zeros(n_analyses)
        upper_bounds = np.zeros(n_analyses)

        while not (
            # monotonic
            fmt_bd.check_monotonicity(
                bounds = np.concatenate((upper_bounds, lower_bounds)),
                n_analyses = n_analyses) and

            # first bounds between -8 and 8
            (lower_bounds[0] > -8) and
            (upper_bounds[0] < 8)
        ):

            # first set of bounds
            # lower_bounds[0] = stats.gennorm.rvs(size=1, beta=0.388, loc=-2.37, scale=0.084)[0]
            # lower_bounds[0] = -1*stats.lognorm.rvs(size=1, s=0.5175, loc=-1.765, scale=3.98)[0]
            # lower_bounds[0] = -1*stats.lognorm.rvs(size=1, s=0.6189, loc=-1.185, scale=3.04)[0]
            lower_bounds[0] = -1*stats.truncnorm.rvs(size=1, a=-0.20909, b=2.544, loc=0.063, scale=3.364)[0]
            upper_bounds[0] = stats.lognorm.rvs(size=1, s=1.12, loc=1.6, scale=1.31)[0]

            for _i in range(n_analyses-1):
                lower_bounds[_i+1] = lower_bounds[_i] + stats.expon.rvs(size=1, loc=0.0488, scale=0.693)[0]
                upper_bounds[_i+1] = upper_bounds[_i] - stats.expon.rvs(size=1, loc=0.000502, scale=0.27)[0]

        return [
            upper_bounds,
            lower_bounds
        ]

    return (boundary_generator,)


@app.cell
def _(boundary_generator):
    boundary_generator(n_analyses=3)
    return


@app.cell
def _(boundary_generator):
    boundary_list = []
    for _i in range(5000):
        boundary_list.append(boundary_generator(n_analyses=3))
    return (boundary_list,)


@app.cell
def _(boundary_list, np, of_bounds, plt, pocock_bounds, tri_bounds):
    _fig, _ax = plt.subplots(figsize=(12,8))

    for _bounds in boundary_list:
        _ax.plot([1,2,3], _bounds[0], color = "purple", alpha=0.05)
        _ax.plot([1,2,3], np.concatenate((_bounds[1][0:2], [_bounds[0][2]])), 
                 color = "purple", alpha=0.05)

    _ax.plot([1,2,3], tri_bounds[0:3], color = "red")
    _ax.plot([1,2,3], tri_bounds[3:7], color = "red")

    _ax.plot([1,2,3], pocock_bounds[0:3], color = "green")
    _ax.plot([1,2,3], pocock_bounds[3:7], color = "green")

    _ax.plot([1,2,3], of_bounds[0:3], color = "blue")
    _ax.plot([1,2,3], of_bounds[3:7], color = "blue")

    _fig
    return


@app.cell
def _(boundary_list):
    index_0 = []
    for _bounds in boundary_list:
        index_0.append(
            (_bounds[1][0]>=-0.4) & (_bounds[0][0] <= 2) & (_bounds[0][2] >= 1.7)
        )
    return (index_0,)


@app.cell
def _(boundary_list, index_0, np):
    lower_0 = np.array(boundary_list)[np.array(index_0)]
    return (lower_0,)


@app.cell
def _(lower_0):
    lower_0.shape
    return


@app.cell
def _():
    160/3000
    return


@app.cell
def _(lower_0, np, of_bounds, plt, pocock_bounds, tri_bounds):
    _fig, _ax = plt.subplots(figsize=(12,8))

    for _bounds in lower_0:
        _ax.plot([1,2,3], _bounds[0], color = "purple", alpha=0.05)
        _ax.plot([1,2,3], np.concatenate((_bounds[1][0:2], [_bounds[0][2]])), 
                 color = "purple", alpha=0.05)

    _ax.plot([1,2,3], tri_bounds[0:3], color = "red")
    _ax.plot([1,2,3], tri_bounds[3:7], color = "red")

    _ax.plot([1,2,3], pocock_bounds[0:3], color = "green")
    _ax.plot([1,2,3], pocock_bounds[3:7], color = "green")

    _ax.plot([1,2,3], of_bounds[0:3], color = "blue")
    _ax.plot([1,2,3], of_bounds[3:7], color = "blue")

    _fig
    return


@app.cell
def _(boundary_generator, np):
    boundaries = []
    for _i in np.arange(start=1, stop=50):
        for _j in np.arange(start=1, stop=21):
            boundaries.append(boundary_generator(n_analyses=_j))
    return (boundaries,)


@app.cell
def _(np):
    xx1 = np.linspace(start=-8, stop=8, num=1000)
    return (xx1,)


@app.cell
def _(
    boundaries,
    lower_bounds_collapsed,
    np,
    plt,
    stats,
    upper_bounds_collapsed,
    xx1,
):
    plt.hist(np.concatenate([np.concatenate(_bound) for _bound in boundaries]), bins=50, density=True, alpha=0.3)
    plt.hist(np.concatenate((upper_bounds_collapsed, lower_bounds_collapsed)), bins=50, density=True, alpha=0.3)
    plt.plot(xx1, stats.lognorm.pdf(x=xx1, s=1.12, loc=1.6, scale=1.31))
    plt.plot(-xx1, stats.truncnorm.pdf(x=xx1, a=-0.20909, b=2.544, loc=0.063, scale=3.364))
    #plt.hist(np.concatenate([np.concatenate(_bound) for _bound in boundary_list]), bins=50, density=True, alpha=0.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## More complex generator from boundary distribution
    """)
    return


@app.cell
def _(fmt_bd, np, stats):
    def boundary_generator2(n_analyses):
        lower_bounds = np.repeat(np.nan, repeats=n_analyses)
        upper_bounds = np.repeat(np.nan, repeats=n_analyses)

        # upper bound distribution
        d1 = stats.lognorm(s=0.7034079900167564, loc=1.6499732459827996, scale=0.6620695872701784)

        # lower bound distribution mixture
        d2 = stats.weibull_max(c=1.4223189831377416, loc=2.4206529325505444, scale=1.6074642047246706)
        d3 = stats.lognorm(s=0.8489881767032773, loc=1.8152781479347766, scale=0.5959923139028731)

        weights = [0.4, 0.6]

        while not fmt_bd.check_monotonicity(
            bounds=np.concatenate((upper_bounds, lower_bounds)),
            n_analyses=n_analyses
        ):

            _i=0
            _j=0

            while _i < n_analyses:
                if _i == 0:
                    upper_bounds[_i] = d1.rvs(size=1)[0]
                    _i += 1
                    continue
    
                temp = d1.rvs(size=1)[0]
            
                if temp > upper_bounds[_i-1]:
                    continue
                else:
                    upper_bounds[_i] = temp
                    _i += 1
    
            while _j < n_analyses:
                choices = np.random.choice([0, 1], size=1, p=weights)
            
                if _j == 0:
                    lower_bounds[_j] = np.select(
                        [choices == 0, choices == 1],
                        [d2.rvs(size=1), d3.rvs(size=1)*-1]
                    )[0]
                    _j += 1
                    continue
            
                temp = np.select(
                    [choices == 0, choices == 1],
                    [d2.rvs(size=1), d3.rvs(size=1)*-1]
                )[0]
            
                if temp < lower_bounds[_j-1]:
                    continue
                else:
                    lower_bounds[_j] = temp
                    _j += 1

        return [
            upper_bounds,
            lower_bounds
        ]

    return (boundary_generator2,)


@app.cell
def _(boundary_generator2):
    boundary_generator2(6)
    return


@app.cell
def _(boundary_generator2):
    boundary_list2 = []
    for _i in range(5000):
        boundary_list2.append(boundary_generator2(n_analyses=3))
    return (boundary_list2,)


@app.cell
def _(boundary_list2, np, of_bounds, plt, pocock_bounds, tri_bounds):
    _fig, _ax = plt.subplots(figsize=(12,8))

    for _bounds in boundary_list2:
        _ax.plot([1,2,3], _bounds[0], color = "purple", alpha=0.05)
        _ax.plot([1,2,3], np.concatenate((_bounds[1][0:2], [_bounds[0][2]])), 
                 color = "purple", alpha=0.05)

    _ax.plot([1,2,3], tri_bounds[0:3], color = "red")
    _ax.plot([1,2,3], tri_bounds[3:7], color = "red")

    _ax.plot([1,2,3], pocock_bounds[0:3], color = "green")
    _ax.plot([1,2,3], pocock_bounds[3:7], color = "green")

    _ax.plot([1,2,3], of_bounds[0:3], color = "blue")
    _ax.plot([1,2,3], of_bounds[3:7], color = "blue")

    _fig
    return


@app.cell
def _(d2, d3, np):
    n1 = 10000

    weights1 = [0.4, 0.6]
    choices1 = np.random.choice([0, 1], size=n1, p=weights1)

    samples1 = np.select(
        [choices1 == 0, choices1 == 1],
        [d2.rvs(size=n1), d3.rvs(size=n1)*-1]
    )
    return (samples1,)


@app.cell
def _(lower_bounds_collapsed, np, plt, samples1, stats):
    _xx = np.linspace(lower_bounds_collapsed.min(), lower_bounds_collapsed.max(), 500)

    _kde = stats.gaussian_kde(samples1, bw_method=0.1)

    plt.hist(lower_bounds_collapsed, bins=75, density=True)
    plt.plot(_xx, _kde(_xx))
    return


@app.cell
def _(boundary_generator2, np):
    boundaries2 = []
    for _i in np.arange(start=1, stop=50):
        for _j in np.arange(start=1, stop=7):
            boundaries2.append(boundary_generator2(n_analyses=_j))
    return (boundaries2,)


@app.cell
def _(boundaries2):
    boundaries2
    return


@app.cell
def _(boundaries2, lower_bounds_collapsed, np, plt, upper_bounds_collapsed):
    plt.hist(np.concatenate([np.concatenate(_bound) for _bound in boundaries2]), bins=50, density=True, alpha=0.3, color="red")
    plt.hist(np.concatenate((upper_bounds_collapsed, lower_bounds_collapsed)), bins=50, density=True, alpha=0.3)
    #plt.hist(np.concatenate([np.concatenate(_bound) for _bound in boundary_list]), bins=50, density=True, alpha=0.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boundary generator using sampling
    """)
    return


@app.cell
def _(d1, d3, np, stats):
    # generate the boundary distribution samples

    # d1 = stats.lognorm(s=0.7034079900167564, loc=1.6499732459827996, scale=0.6620695872701784)
    # d2 = stats.weibull_max(c=1.4223189831377416, loc=2.4206529325505444, scale=1.6074642047246706)
    d2_1 = stats.weibull_max(c=1.4223189831377416, loc=2.4206529325505444, scale=2.5)
    # d3 = stats.lognorm(s=0.8489881767032773, loc=1.8152781479347766, scale=0.5959923139028731)

    num_samples = 250000

    weights_lower = [0.6, 0.4]
    large_choices_lower = np.random.choice([0, 1], size=num_samples, p=weights_lower)

    large_samples_lower = np.select(
        [large_choices_lower == 0, large_choices_lower == 1],
        [d2_1.rvs(size=num_samples), d3.rvs(size=num_samples)*-1]
    )

    large_samples_upper = d1.rvs(size=250000)
    return large_samples_lower, large_samples_upper


@app.cell
def _(
    large_samples_lower,
    large_samples_upper,
    lower_bounds_collapsed,
    np,
    plt,
    upper_bounds_collapsed,
):
    plt.hist(
        np.concatenate((large_samples_lower[large_samples_lower>-9], large_samples_upper[large_samples_upper<9])), 
        density=True, 
        bins=100
    )

    plt.hist(np.concatenate((upper_bounds_collapsed, lower_bounds_collapsed)), bins=50, density=True, alpha=0.3)
    return


@app.cell
def _(np):
    rng = np.random.default_rng()
    return (rng,)


@app.cell
def _(large_samples_lower, large_samples_upper):
    upper_bounds_allowed = large_samples_upper[(large_samples_upper > 1.5) & (large_samples_upper < 9)]
    lower_bounds_allowed = large_samples_lower[(large_samples_lower > -9)  & (large_samples_lower < 2.5)]
    return lower_bounds_allowed, upper_bounds_allowed


@app.cell
def _(fmt_bd, lower_bounds_allowed, np, rng, upper_bounds_allowed):
    def boundary_generator3(n_analyses):
        lower_bounds = np.zeros(n_analyses)
        upper_bounds = np.zeros(n_analyses)

        while not fmt_bd.check_monotonicity(
            bounds=np.concatenate((upper_bounds, lower_bounds)),
            n_analyses=n_analyses
        ):
            # upper bounds generation
            for _i in range(n_analyses):
                if _i == 0: 
                    upper_bounds[_i] = rng.choice(upper_bounds_allowed)
                else:
                    filter = upper_bounds_allowed <= upper_bounds[_i-1]
                    upper_bounds[_i] = rng.choice(upper_bounds_allowed[filter])

            # lower bounds generation
            for _j in range(n_analyses):
                if _j == 0: 
                    lower_bounds[_j] = rng.choice(lower_bounds_allowed)
                else:
                    filter = lower_bounds_allowed >= lower_bounds[_j-1]
                    lower_bounds[_j] = rng.choice(lower_bounds_allowed[filter])

        return [
            upper_bounds,
            lower_bounds
        ]

    return (boundary_generator3,)


@app.cell
def _(boundary_generator3):
    boundary_generator3(3)
    return


@app.cell
def _(boundary_generator3):
    boundary_list3 = []
    for _i in range(5000):
        boundary_list3.append(boundary_generator3(n_analyses=3))
    return (boundary_list3,)


@app.cell
def _(boundary_list3, np, of_bounds, plt, pocock_bounds, tri_bounds):
    _fig, _ax = plt.subplots(figsize=(12,8))

    for _bounds in boundary_list3:
        _ax.plot([1,2,3], _bounds[0], color = "purple", alpha=0.05)
        _ax.plot([1,2,3], np.concatenate((_bounds[1][0:2], [_bounds[0][2]])), 
                 color = "purple", alpha=0.05)

    _ax.plot([1,2,3], tri_bounds[0:3], color = "red")
    _ax.plot([1,2,3], tri_bounds[3:7], color = "red")

    _ax.plot([1,2,3], pocock_bounds[0:3], color = "green")
    _ax.plot([1,2,3], pocock_bounds[3:7], color = "green")

    _ax.plot([1,2,3], of_bounds[0:3], color = "blue")
    _ax.plot([1,2,3], of_bounds[3:7], color = "blue")

    _fig
    return


@app.cell
def _(boundary_list3, np):
    index_0_new = []
    for _bounds in boundary_list3:
        index_0_new.append(
            (_bounds[1][0] >= -0.4) & (_bounds[1][0] <= 0.4) & (_bounds[0][0] <= 2.4) & (_bounds[0][2] >= 1.8)
        )

    lower_0_new = np.array(boundary_list3)[np.array(index_0_new)]

    lower_0_new.shape
    return (lower_0_new,)


@app.cell
def _(lower_0_new, np, of_bounds, plt, pocock_bounds, tri_bounds):
    _fig, _ax = plt.subplots(figsize=(12,8))

    for _bounds in lower_0_new:
        _ax.plot([1,2,3], _bounds[0], color = "purple", alpha=0.05)
        _ax.plot([1,2,3], np.concatenate((_bounds[1][0:2], [_bounds[0][2]])), 
                 color = "purple", alpha=0.05)

    _ax.plot([1,2,3], tri_bounds[0:3], color = "red")
    _ax.plot([1,2,3], tri_bounds[3:7], color = "red")

    _ax.plot([1,2,3], pocock_bounds[0:3], color = "green")
    _ax.plot([1,2,3], pocock_bounds[3:7], color = "green")

    _ax.plot([1,2,3], of_bounds[0:3], color = "blue")
    _ax.plot([1,2,3], of_bounds[3:7], color = "blue")

    _fig
    return


@app.cell
def _(d2):
    d2
    return


@app.cell
def _():
    # d1 = stats.lognorm(s=0.7034079900167564, loc=1.6499732459827996, scale=0.6620695872701784)
    # d2 = stats.weibull_max(c=1.4223189831377416, loc=2.4206529325505444, scale=1.6074642047246706)
    # d3 = stats.lognorm(s=0.8489881767032773, loc=1.8152781479347766, scale=0.5959923139028731)
    return


@app.cell
def _(stats):
    stats.weibull_max.cdf(x=0, c=1.4223189831377416, loc=2.4206529325505444, scale=2.5) -stats.weibull_max.cdf(x=-2, c=1.4223189831377416, loc=2.4206529325505444, scale=2.5)
    return


@app.cell
def _(plt, stats, xx1):
    plt.plot(xx1, stats.weibull_max.pdf(x=xx1, c=1.2, loc=2.4206529325505444, scale=2.5))
    return


@app.cell
def _(np, upper_bounds):
    upper_bound_save = np.array(upper_bounds, dtype=object)
    return (upper_bound_save,)


@app.cell
def _(np, upper_bound_save):
    np.save("/tf/2026-03-t17/upper_bounds.npy", upper_bound_save, allow_pickle=True)
    return


@app.cell
def _(lower_bounds, np):
    lower_bound_save = np.array(lower_bounds, dtype=object)
    return (lower_bound_save,)


@app.cell
def _(lower_bound_save, np):
    np.save("/tf/2026-03-t17/lower_bounds.npy", lower_bound_save, allow_pickle=True)
    return


if __name__ == "__main__":
    app.run()
