import numpy as np
from matplotlib import pyplot as plt
import copy
import json
from tkinter import *
import tkinter.filedialog
import tkinter.messagebox
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.rc('font', size = 12)
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')
mylabelsize = 14
plt.rc('ytick', labelsize=mylabelsize)
plt.rc('xtick', labelsize=mylabelsize)
plt.rc('axes', labelsize=mylabelsize)

plt.close("all")
root = Tk()
root.withdraw()
plt.ion()


def add_breaks_to_length(assign_top_depth=False):
    """

    """

    run_dict_filepath = tkinter.filedialog.askopenfilename(
        initialdir="../", title="Open run json file"
    )
    f = open(run_dict_filepath, "r")
    run_dict = json.load(f)
    f.close()

    encoder_length_filepath = tkinter.filedialog.askopenfilename(
        initialdir="../", title="Open encoder out file"
    )
    print("Opening encoder lenght file: %s" % encoder_length_filepath)
    length_data = np.genfromtxt(encoder_length_filepath, comments="#")
    encoder_secs = length_data[1:, 0]
    encoder_epoch = length_data[1:, 1]
    encoder_length = length_data[1:, 2]

    length_with_breaks = copy.deepcopy(encoder_length)

    # does not include the very last break - ice removed at the end of the run
    for i in range(len(run_dict["break_positions_pre_cut"]) - 1):
        print(
            "Adding break at %0.1f with width of %0.1f" % (
                run_dict["break_positions_pre_cut"][i],
                run_dict["break_lengths"][i]
            ))
        length_with_breaks[length_with_breaks >= run_dict["break_positions_pre_cut"][i]] += run_dict["break_lengths"][i]

    true_depth = copy.deepcopy(length_with_breaks)
    if assign_top_depth:
        run_top_depth = float(input("Give true top depth in meters: ")) * 1000  # convert to mm
        true_depth += run_top_depth


    dataout = np.transpose(np.vstack((encoder_secs, encoder_epoch, encoder_length, length_with_breaks, true_depth)))
    f_name_out = run_dict["core"] + "_run_" + run_dict["run_nr"] + ".brk"
    fileout = tkinter.filedialog.asksaveasfilename(initialdir="../", \
        initialfile = f_name_out, defaultextension = ".brk", title = "Save brk file")
    f = open(fileout, "w")
    f.write("Acquisition_time,Epoch_time,length_melting,length_with_breaks,true_depth\n")
    np.savetxt(f, dataout, fmt = "%i,%0.0f,%0.3f,%0.3f,%0.3f")
    f.close()

    fig1, ax1 = plt.subplots(ncols = 1, nrows = 1, num = 1054, figsize = (10,6), tight_layout = True)
    ax1.plot(encoder_secs, encoder_length, linewidth = 1, color = "k", label = "without breaks")
    ax1.plot(encoder_secs, length_with_breaks, "r", linewidth = 1, label = "with breaks")
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('length [mm]')
    ax1.set_title("Core: %s, Run: %s, Bags %s-%s" %(run_dict["core"], \
        run_dict["run_nr"], run_dict["top_bag"], run_dict["bottom_bag"]))
    ax1.legend(frameon = False, fontsize = 14)



    return

add_breaks_to_length(assign_top_depth = True)
