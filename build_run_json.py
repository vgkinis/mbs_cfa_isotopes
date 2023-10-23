import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import os.path
import json
from tkinter import *
import tkinter.filedialog
import tkinter.messagebox

plt.close("all")
root = Tk()
root.withdraw()
plt.ion()


def interface_user():
    run_dict = {}
    run_dict["run_nr"] = input("Give run nr: ")
    run_dict["run_date"] = input("Give run date YYYYMMDD: ")
    run_dict["top_bag"] = input("Give top bag nr: ")
    run_dict["bottom_bag"] = input("Give bottom bag nr: ")

    run_dict = check_total_length(run_dict)
    run_dict = collect_break_positions(run_dict)

    f_json = open(
        "./" + str(run_dict["core"]) + "_run_" + str(run_dict["run_nr"] + ".json"),
        "w"
    )
    json.dump(run_dict, f_json, sort_keys=True, indent=4)
    f_json.close()
    draw_core(run_dict)
    return


def check_total_length(run_dict):
    bags_in_run = np.arange(int(run_dict["top_bag"]), int(run_dict["bottom_bag"])+1)
    run_dict["cut_files_folder"] = tkinter.filedialog.askdirectory(
        initialdir="../", title="Choose cut files folder"
    )
    cut_files_in_run = list(tkinter.filedialog.askopenfilenames(
        initialdir="../", title='Choose cut files of run'
    ))
    print("List of cut files:")
    for j in cut_files_in_run:
        print(os.path.split(j)[-1])
    sort_list = input("Sort list (0,1): ")
    if sort_list:
        cut_files_in_run.sort()
    lengths_pre_cut = []
    lengths_melted = []
    break_lengths = []
    for fi in cut_files_in_run:
        if fi[0] != ".":
            data_in_fi = np.genfromtxt(
                os.path.join(run_dict["cut_files_folder"], fi),
                dtype="S", delimiter=","
            )
            run_dict["core"] = data_in_fi[0, 0].decode()
            data_fi = data_in_fi[1:, :].astype(int)
            lengths_pre_cut.append(data_fi[0, 0])
            lengths_melted.append(data_fi[0, 1])
            break_lengths.append(np.sum(data_fi[1:, 1]))

    print("\n\nChecking length integrity for run %s" % run_dict["run_nr"])
    print("\nTotal length pre cut: %0.1f\n" % np.sum(lengths_pre_cut))
    print("Total length melting: %0.1f\n" % np.sum(lengths_melted))
    print("Total length breaks: %0.1f\n" % np.sum(break_lengths))
    difference = np.sum(lengths_melted) + np.sum(break_lengths) -\
    np.sum(lengths_pre_cut)
    print("Difference: %0.1f\n" % difference)

    run_dict["run_length_melted"] = str(np.sum(lengths_melted))
    run_dict["run_length_pre_cut"] = str(np.sum(lengths_pre_cut))
    run_dict["total_break_lengths"] = str(np.sum(break_lengths))
    run_dict["cut_files_in_run"] = [os.path.split(i)[-1] for i in cut_files_in_run]

    return run_dict


def collect_break_positions(run_dict):
    break_positions_from_top_of_melted_ice = np.array(())
    break_lengths = np.array(())
    offset_preceding_bags = 0
    for cut_file in run_dict["cut_files_in_run"]:
        data_cut = np.genfromtxt(
            os.path.join(run_dict["cut_files_folder"], cut_file),
            delimiter=",", skip_header=1
        )
        break_positions = data_cut[1:, 0]
        break_positions_from_top_of_melted_ice = np.hstack(
            (break_positions_from_top_of_melted_ice,
                break_positions + offset_preceding_bags)
        )
        break_lengths = np.hstack((break_lengths, data_cut[1:, 1]))
        offset_preceding_bags += data_cut[0, 1]

    dublicates_indexes = np.where(
        np.diff(break_positions_from_top_of_melted_ice) < 1e-3)[0]
    for i in dublicates_indexes:
        break_lengths[i] += break_lengths[i + 1]
    break_lengths = np.delete(break_lengths, dublicates_indexes + 1)
    break_positions_from_top_of_melted_ice = np.delete(
        break_positions_from_top_of_melted_ice, dublicates_indexes + 1
    )

    run_dict["break_positions"] = break_positions_from_top_of_melted_ice.tolist()
    run_dict["break_lengths"] = break_lengths.tolist()
    break_positions_pre_cut = np.zeros(np.size(break_positions_from_top_of_melted_ice))
    break_positions_pre_cut[1:] = break_positions_from_top_of_melted_ice[1:] + np.cumsum(break_lengths[:-1])
    run_dict["break_positions_pre_cut"] = break_positions_pre_cut.tolist()

    return run_dict


def draw_core(run_dict):
    offset_x = 0
    offset_y = 400
    fig, axs = plt.subplots(figsize=(8, 4))
    core_cut = matplotlib.patches.Rectangle(
        xy=(offset_x, offset_y), width=int(run_dict["run_length_melted"]),
        height=100, color="k", fill=False
    )
    axs.add_patch(core_cut)
    for pos in run_dict["break_positions"]:
        break_line = matplotlib.patches.FancyArrowPatch(
            posA=(int(pos), offset_y),
            posB=(int(pos), offset_y + 100), arrowstyle="-"
        )
        axs.add_patch(break_line)

    core_pre_cut = matplotlib.patches.Rectangle(
        xy=(offset_x, offset_y + 300),
        width=int(run_dict["run_length_pre_cut"]),
        height=100, color="k", fill=False
    )
    axs.add_patch(core_pre_cut)
    for j in range(np.size(run_dict["break_positions_pre_cut"])):
        ice_removed = matplotlib.patches.Rectangle(
            xy=(offset_x + int(run_dict["break_positions_pre_cut"][j]),
                offset_y + 300), width=int(run_dict["break_lengths"][j]),
            height=100, color="grey", fill=True
        )
        axs.add_patch(ice_removed)

    axs.set_xlim((0, 1.2 * np.float(run_dict["run_length_pre_cut"])))
    axs.set_ylim((200, 1000))
    # axs.draw()
    fig.show()
    title = "Core: %s - Run: %s" % (run_dict["core"], run_dict["run_nr"])
    print(title)
    axs.set_title(label="Core: %s - Run: %s" % (
        run_dict["core"], run_dict["run_nr"])
    )
    return


interface_user()
