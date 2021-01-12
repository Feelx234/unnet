from collections import defaultdict

import graph_tool.all as gt
import graph_tool.draw as gt_draw
import matplotlib.pyplot as plt
import numpy as np

def plot_graph(G):
    """example use G=lin_log_homo_ba_edges_gt(20,3, 0.05,0.0, 10E-10, force_minority_first=True)
    plot_graph(G)"""
    pos = gt.sfdp_layout(G)

    gt.graph_draw(G, pos=pos,
                  vertex_color=G.vertex_properties['minority'],
                  vertex_fill_color=G.vertex_properties['minority'])


def plot_top_places(result, show_std=False, measure="degree", groupby=['homophily'], cmap=None, maxk=None):
    name=f'{measure}_min_places'
    tmp=result.groupby(groupby).agg({name:list})
    #maj_distr=tmp.degree_maj_places.apply(lambda x: np.mean(np.array(x),axis=0))
    #maj_stds=tmp.degree_maj_places.apply(lambda x: np.std(np.array(x),axis=0))
    min_distr=tmp[name].apply(lambda x: np.mean(np.array(shorten_to_same_length(x)),axis=0))
    if show_std:
        min_stds=tmp[name].apply(lambda x: np.std(np.array(x),axis=0))
    else:
        min_stds =[None for _ in min_distr]

    fig = ensure_figures([],1)[0]
    ax = fig.axes[0]
    cmap=ensure_cmap(cmap)


    for h, min_vals, min_std in zip(min_distr.index, min_distr, min_stds):
        label=min_distr.index.name+f"={h}"
        x=np.arange(1,len(min_vals)+1)
        y=min_vals/x
        color=cmap(h)
        if show_std:
            yerr=min_std/x
            if not (maxk is None):
                y=y[:maxk]
                x=x[:maxk]
                yerr = yerr[:maxk]
            ax.errorbar(x,y, yerr=yerr, label=label, color=color)
        else:
            if not (maxk is None):
                y=y[:maxk]
                x=x[:maxk]
            ax.plot(x, y, label=label, color=color)

    if 'minority_fraction' in result.columns:
        ax.plot(x, np.full(x.shape, result['minority_fraction'].iloc[0]), color="k", linestyle='dashed')
    elif 'minority_measured' in result.columns:
        ax.plot(x, np.full(x.shape, result['minority_measured'].iloc[0]), color="k", linestyle='dashed')
    ax.set_title(f"places by {measure}")
    ax.set_xlabel("k")
    ax.set_ylabel("fraction of minority in top k")
    ax.legend(loc='upper right')
    return fig

def shorten_to_same_length(x):
    length=min(map(len, x))
    has_shortened=0
    y=[]
    for arr in x:
        if len(arr)>length:
            has_shortened+=1
        y.append(arr[:length])
    if has_shortened >0:
        print(f"Shortened {has_shortened} arrays")
    return y

def ensure_cmap(cmap):
    if isinstance(cmap, str):
        return plt.get_cmap(cmap)#(0.1)
    if cmap is None:
        return plt.get_cmap()
    return cmap
    

def frac_of_total(x, t="mean"):
    l=[]
    for arr in x:
        l.append(arr[-1])
    if t=="mean":
        return np.mean(l)
    elif t=="std":
        return np.std(l)

def plot_frac(result, show_std=False, measure="degree", total = 1):
    """
    total=2*(n-1)*m
    """
    name=f'{measure}_min_cumsum'
    tmp=result.groupby(['minority_fraction', 'homophily']).agg({name:list})
    #total = 2*(n-1)*m
    #maj_distr=tmp.degree_maj_cumsum.apply(lambda x: np.mean(np.array(x),axis=0))
    #maj_stds=tmp.degree_maj_cumsum.apply(lambda x: np.std(np.array(x),axis=0))
    min_distr=tmp[name].apply(lambda x: frac_of_total(x))
    min_stds=tmp[name].apply(lambda x: frac_of_total(x, "std"))
    
    vals=defaultdict(list)
    stds=defaultdict(list)
    hs=defaultdict(list)
    for (minority, h), min_vals, min_std in zip(min_distr.index, min_distr, min_stds):

        vals[minority].append(min_vals/total)
        hs[minority].append(h)
        stds[minority].append(min_std/total)

    fig=plt.figure()
    ax = fig.axes[0]
    for key in vals:
        x=hs[key]
        y=vals[key]
        yerr=stds[key]
        
        if not show_std:
            ax.plot(x,y, label="min_size="+str(key))
        else:
            ax.errorbar(x,y,yerr=yerr,label="min_size="+str(key))
    ax.grid()

    ax.xlabel("homophily (h)")
    ax.ylabel("fraction of minority total degree")
    ax.legend(loc='lower left')
    return fig

def ensure_figures(figures, number_of_figures):
    if len(figures)==0:
        figures = [plt.Figure() for _ in range(number_of_figures)]
    for fig in figures:
        if len(fig.axes)==0:
            ax = fig.add_subplot()
    return figures


def plot_distr(result,
                measure="degree",
                scale=['log', 'log'],
                figures=[],
                prefix="",
                show_std=True,
                show_figs=True,
                colors=('r', 'b'),
                normalize=False,
                groupby=['homophily'],
                plus_one=False,
                use_threshold=False):
    maj_name=f'{measure}_distr_maj'
    min_name=f'{measure}_distr_min'
    tmp=result.groupby(groupby).agg({maj_name : list, min_name:list})
    maj_distr = tmp[maj_name].apply(lambda x: np.mean(np.array(x),axis=0))
    maj_stds = tmp[maj_name].apply(lambda x: np.std(np.array(x),axis=0))
    min_distr = tmp[min_name].apply(lambda x: np.mean(np.array(x),axis=0))
    min_stds = tmp[min_name].apply(lambda x: np.std(np.array(x),axis=0))
    
    figures = ensure_figures(figures, len(maj_distr))

    for h, min_vals, min_std, maj_vals, maj_std, fig in zip(maj_distr.index, min_distr, min_stds, maj_distr, maj_stds, figures):
        s1=1#sum(min_vals)
        min_vals/=s1
        min_std/=s1
        s2=1#sum(maj_vals)
        maj_vals/=s2
        maj_std/=s2
        x=np.arange(len(min_vals))
        if use_threshold:
            inds1=min_vals>1
            x=x[inds1]
            min_vals=min_vals[inds1]
            min_std=min_std[inds1]

        x2=np.arange(len(maj_vals))
        if use_threshold:
            inds2=maj_vals>1
            x2=x2[inds2]
            maj_vals=maj_vals[inds2]
            maj_std=maj_std[inds2]
        ax = fig.axes[0]
        ax.set_xscale(scale[0])
        ax.set_yscale(scale[1])

        if plus_one:
            x+=1
            x2+=1

        if normalize:
            f_min = np.sum(min_vals)
            f_maj = np.sum(maj_vals)
            min_vals/= f_min
            maj_vals/= f_maj
            min_std /= min_std
            maj_std /= maj_std

        if show_std:
            ax.errorbar(x,min_vals, yerr=min_std, label=prefix + "min", color=colors[1])
            
            ax.errorbar(x2,maj_vals,yerr=maj_std, label=prefix + "maj", color=colors[0])
        else:    
            ax.plot(x2,maj_vals, label=prefix + "maj", color=colors[0])
            ax.plot(x,min_vals,label=prefix + "min", color=colors[1])
        if plus_one:
            ax.set_xlabel(f"{measure} + 1")
        else:    
            ax.set_xlabel(f"{measure}")
        ax.set_ylabel(f"{measure} distrubution")
        ax.set_title(f"h={h}")
        ax.legend()
    return figures


def cumsum_mean(xs,ys, mode="safe"):
    """properly averages multiple arrays with different x values
    
    It does so by first calculating all possible x_values and then for each array linearly interpolates between the values
    """
    
    # In case there are multiple values for the same x, take the y value corresponding to the first x
    x=[]
    y=[]
    for arr, values in zip(xs,ys):
        inds = np.hstack([True, np.diff(arr)>0])
        x.append(arr[inds])
        y.append(values[inds])
        
    points = np.unique(np.hstack(x))

    
    if mode=="safe":
        mi = max(map(np.min, x))
        ma = min(map(np.max, x))
        filt=np.logical_and(points>=mi, points<=ma)
        points=points[filt]
    
    arrs=[]
    for arr_x, arr_y in zip(x,y):
        #print(arr_x.shape, arr_y.shape)
        arrs.append( np.interp(points, arr_x, arr_y, left=1, right=0))
    arrs = np.array(arrs)
    #print(arrs[:,0])
    #print(x)
    #print(arrs[:,0])
    #print(arrs.shape)
    return points, np.mean(arrs, axis=0), np.std(arrs, axis=0)
    
    
def plot_distr_cumsum(result, measure="degree", scale=['log', 'log'], figures=[], prefix="", show_std=True, show_figs=True, mode="safe", colors=('r', 'b')):
    """ plots the cummulative distribution functions
    special care has to be taken because averaging these is not trivial in comparison to e.g. degree
    """
    maj_name=f'{measure}_distr_cumsum_maj'
    min_name=f'{measure}_distr_cumsum_min'
    
    maj_x = f'{measure}_distr_cumsum_maj_x'
    min_x = f'{measure}_distr_cumsum_min_x'
    tmp=result.groupby(['homophily']).agg({maj_name : list, min_name:list, min_x:list, maj_x:list})

    
    maj = []
    for x,y in zip(tmp[maj_x], tmp[maj_name]):
        x_out, mean_out, std_out = cumsum_mean(x,y, mode=mode)
        maj.append((x_out, mean_out, std_out))
    
    mino = []
    for x,y in zip(tmp[min_x], tmp[min_name]):
        x_out, mean_out, std_out = cumsum_mean(x,y,mode=mode)
        mino.append((x_out, mean_out, std_out))
        
    if len(figures)==0:
        figures = [plt.Figure() for _ in range(len(tmp.index))]
    for fig in figures:
        if len(fig.axes)==0:
            ax = fig.add_subplot()
    
    for h, (min_xx, min_vals, min_std), (maj_xx, maj_vals, maj_std), fig in zip(tmp.index, maj, mino, figures):
        plt.figure()
        x=min_xx
        x2=maj_xx
        ax = fig.axes[0]
        ax.set_xscale(scale[0])
        ax.set_yscale(scale[1])
        if show_std:
            ax.errorbar(x,min_vals, yerr=min_std, label=prefix + "min", color=colors[0])
            
            ax.errorbar(x2,maj_vals,yerr=maj_std, label=prefix + "maj", color=colors[1])
        else:    
            ax.plot(x,min_vals,label=prefix + "min", color=colors[0])
            ax.plot(x2,maj_vals, label=prefix + "maj", color=colors[1])
            #print(maj_vals)
        ax.set_xlabel(f"{measure}")
        ax.set_ylabel(f"{measure} distrubution")
        ax.set_title(f"h={h}")
        ax.legend()
    return figures