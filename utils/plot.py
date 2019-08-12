import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from scipy.interpolate import make_interp_spline, BSpline

def plot(means,maxes,mins,mean_steps,num_agents=2,name='DDPG',game='Tennis'):
     
    length = len(means)
    means = np.array(means)
    maxes = np.array(maxes)
    mins = np.array(mins)
    mean_steps = np.array(mean_steps)

    xline = np.linspace(0,length,length*10)
    xfit = np.arange(length)
    
    spl = make_interp_spline(xfit,means,k=3)
    spl2 = make_interp_spline(xfit,mins,k=3)
    spl3 = make_interp_spline(xfit,maxes,k=3)
    spl4 = make_interp_spline(xfit,mean_steps,k=3)

    means_smooth = spl(xline)
    mins_smooth = spl2(xline)
    maxes_smooth = spl3(xline)
    steps_smooth = spl4(xline)

    _, ax = plt.subplots()

    title = "{} performance on Unity {} with {} agents".format(name,game,num_agents)
    x_label = "Number of Episodes"
    y_label = "Score"

    line1 = ax.plot(xline, means_smooth, lw=1, color= '#539caf', alpha = 1, label= 'Mean score')
    line3 = ax.fill_between(xline,mins_smooth,maxes_smooth,color='orange',alpha = 0.4, label = 'Min/Max score')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    blue_patch = mpatches.Patch(color='#539caf', label='Mean score')
    orange_patch = mpatches.Patch(color='#FFA500', label='Min/Max score')
    plt.legend(handles=[blue_patch,orange_patch])
    # plt.show()
    plt.savefig(str(name)+'_performance.png',bbox_inches='tight')

    plt.close()

    _, ax1 = plt.subplots()
    ax1.plot(xline, steps_smooth, lw=1, color= '#539caf', alpha = 1, label= 'Mean score')
    ax1.set_title('DDPG mean steps in Unity Tennis')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Mean steps')

    plt.savefig(str(name)+'_mean_steps.png',bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # means, stds = pickle.load(open('maddpg_scores.p', 'rb'))
    # test
    means = np.arange(5)
    mins = np.ones(5)
    maxes = np.arange(5)
    steps = np.ones(5)
    plot(means,maxes,mins,steps)