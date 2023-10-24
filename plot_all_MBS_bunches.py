import sys
sys.path.append("/home/vasileios/")
sys.path.append("/home/vasilis/")
import vaspy
from vaspy import syttensen
import vaspy.syttensen.bunch_syttensen as bunch_syttensen
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import copy
import os
import string
from tkinter import *
import tkinter.filedialog
import tkinter.messagebox
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.rc('font', size = 12)
plt.rc('xtick', direction = 'in')
plt.rc('ytick', direction = 'in')
mylabelsize = 14
plt.rc('ytick', labelsize = mylabelsize)
plt.rc('xtick', labelsize = mylabelsize)
plt.rc('axes', labelsize = mylabelsize)

plt.close("all")
root = Tk()
root.withdraw()
plt.ion()

def combine_bunches():
    dir_to_files = tkinter.filedialog.askdirectory(initialdir = "../")
    super_bunch_switch = input("Save superbunch [Y/n]? ")
    if super_bunch_switch in ["y", "Y", "1"]:
        print("superbunch switch")
        super_bunch = bunch_syttensen.Bunch()
        super_bunch_counter = 0

    list_of_all_files = tkinter.filedialog.askopenfilenames(initialdir = "../", title='Choose run files on depth')
    x_var_input = input("Plot on depth/epoch [0,1]?: ")

    prune_switch = input("Prune top/bottom of runs [Y/n]? ")
    if prune_switch in ["y", "Y", "1"]:
        dz_list = input("Section length top, bottom to be prunned [in mm!]: ").split(",")
        dz_top = np.float(dz_list[0])
        dz_bottom = np.float(dz_list[1])
    else:
        pass

    fig1, ax1 = plt.subplots(ncols = 1, nrows = 1, num = 1, figsize = (10,6), tight_layout = True)
    for j in list_of_all_files:
        print("Checkin file %s" %j)
        b = bunch_syttensen.Bunch()
        b.unpickle_bunch(j)
        if x_var_input == "0":
            x_var = b.z/1000
            label_x = "Depth [m]"
        else:
            x_var = b.epoch
            label_x = "Epoch time [s]"

        ax1.plot(x_var, b.dD, linewidth = 0.7)
        print("plotting %s" %j)

        if prune_switch in ["y", "Y", "1"]:
            z_top_new = np.min(b.z) + dz_top
            z_bottom_new = np.max(b.z) - dz_bottom
            criterion = (b.z > z_top_new) & (b.z < z_bottom_new)
            b = b.pick(np.where(criterion)[0][0], np.where(criterion)[0][-1])
            if x_var_input == "0":
                x_var_pruned = b.z/1000
            else:
                x_var_pruned = b.epoch
            ax1.plot(x_var_pruned, b.dD, linewidth = 0.7)
            print("plotting %s" %j)

            if super_bunch_switch in ["y", "Y", "1"]:
                if super_bunch_counter == 0:
                    super_bunch = copy.deepcopy(b)
                    super_bunch_counter +=1
                else:
                    super_bunch.concat(b)
        else:
            continue
    ax1.set_xlabel(label_x)
    ax1.set_ylabel("$\delta\mathrm{D}$")

    sort_indexes = np.argsort(super_bunch.z)
    for i in super_bunch.__dict__.keys():
        try:
            super_bunch.__dict__[i] = super_bunch.__dict__[i][sort_indexes]
        except:
            print("Error sorting %s" %i)
    print(super_bunch.secs)

    if super_bunch_switch in ["y", "Y", 1]:
        fig2, ax2 = plt.subplots(ncols = 1, nrows = 1, num = 2, figsize = (10,6), tight_layout = True)
        ax2.plot(super_bunch.z/1000, super_bunch.dD, linewidth = 0.7, color = "firebrick")
        ax2.set_xlabel("Depth [m]")
        ax2.set_ylabel("$\delta\mathrm{D}$")

        super_bunch_out_filepath = tkinter.filedialog.asksaveasfilename(initialdir = "../")
        super_bunch.pickle_bunch(super_bunch_out_filepath+".gil")
        super_bunch.save_ascii(super_bunch_out_filepath+".txt", z = True)
        fig3, ax3 = plt.subplots(ncols = 1, nrows = 1, num = 5, figsize = (10,6), tight_layout = True)
        ax3.plot(super_bunch.epoch, super_bunch.z, linewidth = 0.7, color = "firebrick")
        ax3.set_xlabel("Measurement time [s]")
        ax3.set_ylabel("Depth [mm]")


    return


combine_bunches()
