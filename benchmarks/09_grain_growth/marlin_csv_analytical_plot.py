#This plots data from a Marlin .csv output file for the shrinking grain
#benchmark problem and compares to the analytical model
import numpy as np
import matplotlib.pyplot as plt
import csv

file_base = '9a_bicrystal_out'
variable = 'areas_diff';

# Extract data
with open(file_base + '.csv','r') as csv_file:
    data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)
times = data[:, 0]
var = data[:, 1]

r0 = 400 #nm
Mgamma = 180 #nm^2/microsec
times_an = np.linspace(0,500, num=101)
area_an = np.pi * r0**2 - 2*np.pi*Mgamma*times_an
area_an[90:] = 0 #Set to 0 after grain shrinks to radius 0

diff = var - area_an

# Example of use
#x_sample = np.array([i*3.14*2/50 for i in range(50)])
#x_vects = [x_sample, x_sample, x_sample, x_sample, x_sample, x_sample]
#y_vects = [np.cos(x_sample),np.cos(x_sample+0.1),np.cos(x_sample+0.2),np.sin(x_sample-2),np.sin(x_sample-2.1),np.sin(x_sample-2.2)]
x_vects = [times_an]
y_vects = [diff]
n_fig = 1
title = ''
xlabel = r'Time ($\mu$s)'
ylabel = r'Difference (nm$^2$)'
axis_range = [0,500,0,6e5]
legend = [r'Difference']
filename = variable
color = '#1f77b4'


plt.rcParams.update({
    "text.usetex":True, # Use Latex rendering for figures
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Sans Serif"]})

# Main function
def plot_profile(x_vects, y_vects, n_fig, title, xlabel, ylabel, axis_range, filename, color, legend):
    ## Pots the profiles in x_vects as functions of y_labels##

    # x_vects contains the x values of the profiles
    # y_vects contains the y values of the profiles
    # n_fig is the figure number # can be useful whenusing this function in a for loop to distinguish between figures
    # title is the title of the figure
    # xlabel_title is the label for the x axis
    # ylabel_title is the label for the y axis
    # axis_range is the range of the two axis [xmin, xmax, ymin, ymax]
    # filename is the name of the saved figure
    # color is the color of the profiles

    # Create figure
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(1,1,1)

    # Plot figure
    for i in range(len(x_vects)):
        # plt.plot(x_vects[i],y_vects[i],ls='-',alpha=0.3,c=color)
        plt.plot(x_vects[i],y_vects[i],'o')


    ax.tick_params(axis='both', direction='in')
    # Complete figure
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    # plt.axis(axis_range)
    # plt.legend(legend, fontsize=12)
    plt.tight_layout()

    ## Save pdf and png files
    plt.savefig(filename + '.pdf')
    plt.savefig(filename + '.png', dpi=200)

plot_profile(x_vects, y_vects, n_fig, title, xlabel, ylabel, axis_range, filename, color, legend)
