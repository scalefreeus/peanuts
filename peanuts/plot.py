import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import random
import numpy as np

plt.rcParams["figure.figsize"] = (20,4)

def scatterplot(x_data, y_data, x_label="", y_label="", title="", color="r", yscale_log=False):

    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    ax.scatter(x_data, y_data, s=10, color=color, alpha = 0.75)

    if yscale_log == True:
        ax.set_yscale('log')

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def linechart(x_data, y_data, x_label="", y_label="", title=""):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax.plot(x_data, y_data, lw=2, color='#bc5090', alpha=1)

    # label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def histogram(data, n_bins = 10, cumulative=False, x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    ax.hist(data, bins = n_bins, cumulative = cumulative, color='#003f5c')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)


# Overlay 2 histograms to compare them
def histogram_overlaid(data1, data2, n_bins = 0, data1_name="1", data1_color="#003f5c", data2_name="2", data2_color="#ffa600", x_label="", y_label="", title=""):
    # Set the bounds for the bins so that the two distributions are fairly compared
    max_nbins = 10
    data_range = [min(min(data1), min(data2)), max(max(data1), max(data2))]
    binwidth = (data_range[1] - data_range[0]) / max_nbins

    if n_bins == 0:
        bins = np.arange(data_range[0], data_range[1] + binwidth, binwidth)
    else:
        bins = n_bins

    # Create the plot
    _, ax = plt.subplots()
    ax.hist(data1, bins = bins, color = data1_color, alpha = 1, label = data1_name)
    ax.hist(data2, bins = bins, color = data2_color, alpha = 0.75, label = data2_name)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc = 'best')

def barchart(x_data, y_data, error_data=None, x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    # Draw bars, position them in the center of the tick mark on the x-axis
    ax.bar(x_data, y_data, color = '#539caf', align = 'center')
    # Draw error bars to show standard deviation, set ls to 'none'
    # to remove line between points
    if error_data != None:
        ax.errorbar(x_data, y_data, yerr = error_data, color = '#297083', ls = 'none', lw = 2, capthick = 2)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)

def barchart_stacked(x_data, y_data_list, y_data_names=None, colors=None, x_label="", y_label="", title=""):
    _, ax = plt.subplots()

    # Draw bars, one category at a time
    for i in range(0, len(y_data_list)):
        if (colors == None):
            color = "#%06x" % random.randint(42, 0xFFFFFF)
        else:
            color = colors[i]

        if y_data_names == None:
            y_data_name = "#{i}".format(i=i+1)
        else:
            y_data_name = y_data_names[i]

        if i == 0:
            ax.bar(x_data, y_data_list[i], color = color, align='center', label = y_data_name)
        else:
            # For each category after the first, the bottom of the bar
            # will be the top of the last category
            ax.bar(x_data, y_data_list[i], color = color, bottom = y_data_list[i - 1], align='center', label = y_data_name)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc = 'upper right')

def barchart_grouped(x_data, y_data_list, y_data_names=None, colors=None, x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    # Total width for all bars at one x location
    total_width = 0.8
    # the x locations for the groups
    ind = np.arange(len(y_data_list[0]))
    # Width of each individual bar
    ind_width = total_width / len(y_data_list)
    # This centers each cluster of bars about the x tick mark
    alteration = np.arange(-(total_width/2), total_width/2 , ind_width)

    # Draw bars, one category at a time
    for i in range(0, len(y_data_list)):
        if (colors == None):
            color = "#%06x" % random.randint(42, 0xFFFFFF)
        else:
            color = colors[i]

        if y_data_names == None:
            y_data_name = "#{i}".format(i=i+1)
        else:
            y_data_name = y_data_names[i]

        # Move the bar to the right on the x-axis so it doesn't
        # overlap with previously drawn ones
        ax.bar(ind + alteration[i], y_data_list[i], width=ind_width, color = color, align='center', label=y_data_name)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.set_xticks(ind - ind_width/2)
    ax.set_xticklabels(x_data)
    ax.legend(loc = 'upper right')

def boxplot(x_data, y_data, base_color="#539caf", median_color="#297083", x_label="", y_label="", title=""):
    _, ax = plt.subplots()

    # Draw boxplots, specifying desired style
    ax.boxplot(y_data
               # patch_artist must be True to control box fill
               , patch_artist = True
               # Properties of median line
               , medianprops = {'color': median_color}
               # Properties of box
               , boxprops = {'color': base_color, 'facecolor': base_color}
               # Properties of whiskers
               , whiskerprops = {'color': base_color}
               # Properties of whisker caps
               , capprops = {'color': base_color})

    # By default, the tick label starts at 1 and increments by 1 for
    # each box drawn. This sets the labels to the ones we want
    ax.set_xticklabels(x_data)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
