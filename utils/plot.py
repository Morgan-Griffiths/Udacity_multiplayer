import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import make_interp_spline, BSpline

def plot(means,stds,num_agents=2,name='DDPG',game='Tennis'):
     
    length = len(means)
    means = np.array(means)
    stds = np.array(stds)

    mins = means-stds
    maxes = means+stds

    xline = np.linspace(0,length,length*10)
    xfit = np.arange(length)
    
    spl = make_interp_spline(xfit,means,k=3)
    spl2 = make_interp_spline(xfit,mins,k=3)
    spl3 = make_interp_spline(xfit,maxes,k=3)

    means_smooth = spl(xline)
    mins_smooth = spl2(xline)
    maxes_smooth = spl3(xline)

    _, ax = plt.subplots()

    title = "{} performance on {} with {} agents".format(name,game,num_agents)
    x_label = "Number of Episodes"
    y_label = "Score"

    ax.plot(xline, means_smooth, lw=1, color= '#539caf', alpha = 1, label= 'mean')
    ax.fill_between(xline,mins_smooth,maxes_smooth,color='orange',alpha = 0.4, label = 'Min/Max')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # plt.show()
    plt.savefig(str(name)+'_performance.png',bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # means, stds = pickle.load(open('maddpg_scores.p', 'rb'))
    means = np.arange(5)
    stds = np.ones(5)
    plot(means,stds)