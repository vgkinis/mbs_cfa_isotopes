from tkinter import *
# import ttk
import tkinter.filedialog
import tkinter.messagebox
import sys
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


sys.path.append("/home/guest/")
sys.path.append("/home/vasilis/")
sys.path.append("/Users/vasilis/")
sys.path.append("/home/guest/vaspy/")
sys.path.append("/home/cfalenovo/")

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


def read_dat():
    """

    """
    b = bunch_syttensen.Bunch()

    dat_filepath = input(
        "Give full filepath of dat file ([d, D, dialog] for Tk): "
    )
    if dat_filepath in ["dialog", "d", "D"]:
        dat_filepath = tkinter.filedialog.askopenfilename(initialdir="./")
    else:
        pass

    dat_filename = os.path.split(dat_filepath)[1]
    print(("\nReading data from %s") % dat_filename)
    b.read_data(dat_filepath)

    gil_out_filepath = input(
        "\nGive full filepath for gil file ([d, D, dialog] for Tk): "
    )
    if gil_out_filepath in ["dialog", "d", "D"]:
        gil_out_filepath = tkinter.filedialog.asksaveasfilename(
            initialdir="./",
            initialfile=os.path.splitext(dat_filename)[0],
            defaultextension=".gil"
        )
    else:
        pass
    b.pickle_bunch(gil_out_filepath)
    print(("Pickling gil file: %s" % os.path.split(gil_out_filepath)[1]))

    ##################
    # SMOW/SLAP Block
    ##################
    smow_switch = input("\n\nCalibrate on SMOW/SLAP ? [y/n]: ")
    if smow_switch == "y":
        p17 = input("Slope/Inters for d17: ").split(sep=",")
        p17 = [float(i) for i in p17]
        p18 = input("Slope/Inters for d18: ").split(sep=",")
        p18 = [float(i) for i in p18]
        pD = input("Slope/Inters for dD: ").split(sep=",")
        pD = [float(i) for i in pD]

        b_smow = b.smow(p17, p18, pD)
        gil_smow_filename = os.path.splitext(dat_filename)[0] + "_smow"

        gil_smow_filepath = input(
            "\nGive full filepath for _smow gil file ([d, D, dialog] for Tk): "
        )
        if gil_smow_filepath in ["dialog", "d", "D"]:
            gil_smow_filepath = tkinter.filedialog.asksaveasfilename(
                initialdir="./",
                initialfile=gil_smow_filename,
                defaultextension=".gil"
            )
        else:
            pass
        b_smow.pickle_bunch(gil_smow_filepath)
        print(("Pickling gil file: %s" % os.path.split(gil_smow_filepath)[1]))

    else:
        pass

    block_1_switch = input(
        "1. Rerun Block 1 \n2. Continue with block 2\n3. Exit\n[1, 2, 3] ?: "
    )
    if block_1_switch == "2":
        read_logs()
    elif block_1_switch == "1":
        read_dat()
    else:
        sys.exit()

    return


def read_logs():
    """

    """
    log_filepath = input(
        "\nGive full filepath of logfile ([d, D, dialog] for Tk): "
    )
    if log_filepath in ["dialog", "d", "D"]:
        log_filepath = tkinter.filedialog.askopenfilename(
            initialdir="./"
        )
    else:
        pass

    print(("Reading log file: %s" % os.path.split(log_filepath)[1]))
    log_data = np.loadtxt(log_filepath, skiprows=1, dtype="S10")
    cfa_date = log_data[:, 0]
    cfa_utc_time = log_data[:, 1]
    cfa_epoch_time = log_data[:, 2].astype(float)
    cfa_on_flag = log_data[:, 3].astype(int)
    cfa_top_bag = log_data[:, 4].astype(int)

    gil_filepath = input(
        "\nGive full filepath of GIL file ([d, D, dialog] for Tk): "
    )
    if gil_filepath in ["dialog", "d", "D"]:
        gil_filepath = tkinter.filedialog.askopenfilename(
            initialdir="./"
        )
    else:
        pass
    print(("Unpickling gil file: %s" % os.path.split(gil_filepath)[1]))
    b = bunch_syttensen.Bunch()
    b.unpickle_bunch(gil_filepath)
    # f = open(gil_filepath, "r")
    # b = pickle.load(f)
    # f.close()

    out_filepath = input(
        "Give filepath (up to bag nr.) for output files ([d, D, dialog] for Tk): "
    )
    if out_filepath in ["dialog", "d", "D"]:
        out_filepath = tkinter.filedialog.asksaveasfilename(initialdir="./")
    else:
        pass

    unique_bags = np.unique(cfa_top_bag)
    print(("Top bags of separate runs:", unique_bags))
    print("-----------------\n")
    nr_of_runs = np.size(unique_bags)

    for j in unique_bags:
        print(j)
        crit = (cfa_top_bag == j) & (cfa_on_flag == 1)
        if not np.any(crit):
            continue

        indexes = np.where(crit)[0]
        epoch_1 = cfa_epoch_time[indexes[0]]
        epoch_2 = cfa_epoch_time[indexes[-1]]
        index_1 = np.where(b.epoch > epoch_1)[0][0]
        index_2 = np.where(b.epoch > epoch_2)[0][0]
        b1 = b.pick(index_1, index_2)
        run1 = bunch_syttensen.Run(b1)
        b1.plot()

        filepath_1 = out_filepath + "_" + str(j) + "_bunch.gil"
        b1.pickle_bunch(filepath_1)

        filepath_2 = out_filepath + "_" + str(j) + "_run.gil"
        run1.pickle_bunch(filepath_2)
        print(("Writing %s" % os.path.split(filepath_2)[1]))

        input("press any key to continue with next section: ")

    block_2_switch = input(
        "1. Rerun Block 2 \n2. Continue with block 3\n3. Exit\n[1, 2, 3] ?: "
    )
    if block_2_switch == "1":
        read_logs()
    elif block_2_switch == "2":
        set_on_depth()
    else:
        sys.exit()

    return


def set_on_depth():
    """

    """
    plt.close("all")
    plt.ion()
    # loading bunch
    run_filepath = input(
        "\nGive full filepath of Run GIL file ([d, D, dialog] for Tk): "
    )
    if run_filepath in ["dialog", "d", "D"]:
        run_filepath = tkinter.filedialog.askopenfilename(
            initialdir="../", title="Open Run GIL file"
        )
    else:
        pass
    print(("Unpickling run gil file: %s" % os.path.split(run_filepath)[1]))
    b = bunch_syttensen.Bunch()
    b.unpickle_bunch(run_filepath)
    b = bunch_syttensen.Run(b)

    print("Auto Locating CFA Run.....")
    try:
        b.locate()
    except IndexError:
        print("Auto Locate failed go for manual.")
    locate_message = input("Auto locate (y)?: ")

    if locate_message != "y":
        b = bunch_syttensen.Bunch()
        b.unpickle_bunch(run_filepath)
        b = bunch_syttensen.Run(b)
        smoothed_dD = b.smooth(b.dD, 20, window="bartlett")
        diff_dD = np.diff(smoothed_dD)

        plt.figure(21)
        plt.clf()
        plt.subplot(211)
        plt.plot(b.index_i, smoothed_dD, "b")
        plt.subplot(212)
        plt.plot(b.index_i[:-1], diff_dD, "b")

        locate_indexes = input("Give locate indexes (comma sep):")
        index_i = int(locate_indexes.split(",")[0])
        index_f = int(locate_indexes.split(",")[1])
        b.locate(index_i=index_i, index_f=index_f)
    else:
        pass

    cfa_depth_filepath = input(
        "\nGive full filepath of CFA depth .brk file ([d, D, dialog] for Tk): "
    )
    if cfa_depth_filepath in ["dialog", "d", "D"]:
        cfa_depth_filepath = tkinter.filedialog.askopenfilename(
            initialdir="../",
            title="Give CFA brk file"
        )
    else:
        pass
    print(("Opening %s" % os.path.split(cfa_depth_filepath)[1]))
    depth_cfa_data = np.genfromtxt(
        cfa_depth_filepath,
        skip_header=1,
        delimiter=","
    )
    melt_secs = depth_cfa_data[:, 0]
    melt_depth = depth_cfa_data[:, 4]
    cfa_melting_time = melt_secs[-1] - melt_secs[0]
    iso_melting_time = b.secs[-1] - b.secs[0]
    diff_melting_time = iso_melting_time - cfa_melting_time

    print(("\nTotal melting time from CFA: %0.2f") % cfa_melting_time)
    print(("Total melting time from ISO: %0.2f") % iso_melting_time)
    print(("Difference ISO - CFA: %0.2f") % diff_melting_time)

    # Scaling/tretching/squeezing the cfa time so it matches the iso time
    time_scaling = b.secs[-1] / melt_secs[-1]
    melt_secs_scaled = melt_secs * time_scaling
    print("Total melting time after scaling: %0.2f" % melt_secs_scaled[-1])
    b.interp_depth(melt_secs_scaled, melt_depth)
    plt.figure(31)
    plt.subplot(211)
    plt.plot(b.z, b.dD, "r")
    plt.ylabel("dD")
    plt.subplot(212)
    plt.plot(b.z, b.dD - 8 * b.d18, "g")
    plt.ylabel("Dxs")
    plt.xlabel("Depth")

    out_filepath = input(
        "Give filepath for output files ([d, D, dialog] for Tk): "
    )
    if out_filepath in ["dialog", "d", "D"]:
        init_file = os.path.splitext(
            os.path.split(run_filepath)[1])[0] + "_ondepth"
        out_filepath = tkinter.filedialog.asksaveasfilename(
            initialdir="../",
            initialfile=init_file,
            defaultextension=".gil"
        )
    else:
        pass

    b.pickle_bunch(out_filepath)

    a = input("press:")
    return


def split_depth_multiruns():
    """

    """
    cfa_depth_filepath = input(
        "\nGive full filepath of CFA depth file ([d, D, dialog] for Tk): "
    )
    if cfa_depth_filepath in ["dialog", "d", "D"]:
        cfa_depth_filepath = tkinter.filedialog.askopenfilename(
            initialdir="../"
        )
    else:
        pass
    print(("Opening %s" % os.path.split(cfa_depth_filepath)[1]))
    data_depth = np.genfromtxt(
        cfa_depth_filepath,
        skip_header=1,
        delimiter=",",
        usecols=(0, 1, 2, 3),
        filling_values=-333.
    )

    indexes_melt = np.where(data_depth[:, 2] != -333.)[0]
    melt_secs = data_depth[indexes_melt, 0]
    melt_secs = melt_secs - melt_secs[0]
    melt_epoch = data_depth[indexes_melt, 1]
    melt_length = data_depth[indexes_melt, 2]
    melt_length_breaks = data_depth[indexes_melt, 3]
    melt_true_depth = data_depth[indexes_melt, 4]
    cfa_melting_time = melt_secs[-1] - melt_secs[0]
    print(("Total melting time from ISO: %0.2f") % cfa_melting_time)

    plt.figure(947)
    plt.plot(melt_secs, melt_length)
    plt.xlabel("Time")
    plt.ylabel("Melting Length")

    runs_times_input = input(
        "Give times bracketing the separate runs comma sep: "
    )
    runs_times = list(map(int, runs_times_input.split(",")))
    print(runs_times)

    sep_runs = len(runs_times) / 2
    for runj in range(sep_runs):
        time_in = runs_times[runj * 2]
        time_fin = runs_times[runj * 2 + 1]
        print((
            "Processing run from %i to %i sec" % (time_in, time_fin)
        ))
        index_1 = np.where(np.abs(melt_secs - time_in) < 0.01)[0]
        index_2 = np.where(np.abs(melt_secs - time_fin) < 0.01)[0]
        print(index_1, index_2)

        melt_secs_runj = melt_secs[index_1:index_2]
        melt_epoch_runj = melt_epoch[index_1:index_2]
        melt_length_runj = melt_length[index_1:index_2]
        melt_depth_runj = melt_depth[index_1:index_2]
        plt.plot(melt_secs_runj, melt_length_runj, "r")
        accept_input = input("Accept run nr. %i ? [y/n]: " % (runj + 1))
        if accept_input in ["N", "n", "0"]:
            sys.exit()
        else:
            pass
        dataout = np.transpose(
            np.vstack((
                melt_secs_runj,
                melt_epoch_runj,
                melt_length_runj,
                melt_depth_runj
            ))
        )

        out_filepath = input(
            "Give filepath for output files ([d, D, dialog] for Tk): "
        )
        if out_filepath in ["dialog", "d", "D"]:
            init_file = os.path.splitext(
                os.path.split(cfa_depth_filepath)[1])[0] + "_" + str(runj + 1
            )
            out_filepath = tkinter.filedialog.asksaveasfilename(
                initialdir="../",
                initialfile=init_file,
                defaultextension=".txt"
            )
        else:
            pass
        f = open(out_filepath, "w")
        f.write("acquisition_time,epoch_time,depth_melting,depth_ice_sheet\n")
        np.savetxt(f, dataout, fmt="%i,%0.2f,%0.5f,%0.5f")
        f.close()

    return


if __name__ == '__main__':
    plt.ion()
    main_table = "1. Read .dat/picle bunches/SMOW cal"
    main_table += "\n2. Read cfa_logs/create run bunches gil files"
    main_table += "\n3. Locate runs/assign depth\n"
    main_table += "4. Split Renland depth files for multiruns\n"
    print(main_table)
    switch_menu = int(input("Choose program section [1,2,3,4]: "))

    if switch_menu == 1:
        read_dat()
    elif switch_menu == 2:
        read_logs()
    elif switch_menu == 3:
        set_on_depth()
    elif switch_menu == 4:
        split_depth_multiruns()
