"""
#######################################################################################
Changes history for depth_encoder.py
#
#20170712:
#
#######################################################################################
"""

import numpy as np
import scipy
from scipy import signal, ndimage, optimize
import pandas as pd
from matplotlib import pyplot as plt
import time
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


class EncoderProcessor(object):

    def __init__(self):
        """

        """

        return

    def build_params_json(self):
        self.params_dict = {}
        self.params_dict["counts_per_mm"] = 25
        self.params_dict["diff_counts_while_loading"] = 15.4
        self.params_dict["clipping_min"] = 0.5
        self.params_dict["filter_size"] = 11
        self.params_dict["filter_type"] = "hann"
        self.params_dict["_filter_types"] = ["barthann", "bartlett", "boxcar", "hamming", "hann"]

        f_json = open("./depth_encoder_params.json", "w")
        json.dump(self.params_dict, f_json, sort_keys = True, indent = 4)
        f_json.close()

        return

    def read_params_json(self, json_path = "./depth_encoder_params.json"):
        f_json = open(json_path, "r")
        print("Reading %s" %json_path)
        self.params_dict = json.load(f_json)
        for j in self.params_dict.keys():
            print("%s: %s" %(j, str(self.params_dict[j])))
            time.sleep(0.5)
        f_json.close()

        run_dict_filepath = tkinter.filedialog.askopenfilename(initialdir = "../", title = "Open run json file")
        f = open(run_dict_filepath, "r")
        run_dict = json.load(f)
        f.close()
        self.params_dict["core"] = run_dict["core"]
        self.params_dict["run_nr"] = run_dict["run_nr"]
        self.params_dict["top_bag"] = run_dict["top_bag"]
        self.params_dict["bottom_bag"] = run_dict["bottom_bag"]
        self.params_dict["run_length_melted"] = np.float(run_dict["run_length_melted"])

        cfa_file_path = tkinter.filedialog.askopenfilename(initialdir="../", title="Choose cfa file")
        self.params_dict["cfa_file_path"] = cfa_file_path

        return

    def read_cfa_file(self, plot_data=True):
        try:
            self.cfa_df = pd.read_csv(self.params_dict["cfa_file_path"], delimiter="\t")
            print(self.cfa_df)

        except:
            print("CFA data file could not be read")

        self.epoch = self.cfa_df['Epoch_time'].values
        self.secs = self.epoch - self.epoch[0]
        self.sample_on = self.cfa_df['MQ/SPLE'].values
        self.counts = self.cfa_df['Encoder_counts'].values
        self.diff_counts = self.cfa_df['Diff_counts'].values
        self.diff2_counts = np.gradient(self.diff_counts)

        ##Defining conditions for sample On and flagging possible loading positions
        condition_sample = self.sample_onb > 0
        condition_loading = (np.abs(self.diff2_counts) > 20) & (self.sample_on > 0)

        print("Median of diff counts: %0.3f" % np.median(self.diff_counts))

        if plot_data:
            ##Plotting encoder counts raw
            fig1, ax1 = plt.subplots(ncols=1, nrows=1, num=1, figsize=(10,6), tight_layout=True)
            ax1.plot(self.secs, self.counts, linewidth=1, color="k")
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Encoder Counts')
            ax1_1 = ax1.twinx()
            ax1_1.plot(self.secs, self.sample_on, "r", linewidth=1)
            ax1_1.set_ylabel("Sample ON", rotation=270)
            ax1_1.yaxis.set_label_coords(1.06, 0.5)
            ax1.set_title("Core: %s, Run: %s, Bags %s-%s" % (self.params_dict["core"], \
                self.params_dict["run_nr"], self.params_dict["top_bag"], self.params_dict["bottom_bag"]))

            ##Plotting 1st derivative of encoder counts raw
            self.fig2, self.ax2 = plt.subplots(ncols=1, nrows=1, num=2, figsize=(10,6), tight_layout=True)
            self.ax2.plot(self.secs, self.diff_counts, linewidth=1, color="c", label="Raw")
            self.ax2.set_xlabel('Time [s]')
            self.ax2.set_ylabel('Diff-counts')
            self.ax2_1 = self.ax2.twinx()
            self.ax2_1.set_ylabel("Sample ON", rotation=270)
            self.ax2_1.yaxis.set_label_coords(1.06, 0.5)
            self.ax2_1.plot(self.secs, self.sample_on, "r", linewidth=1)
            self.ax2.set_title("Core: %s, Run: %s, Bags %s-%s" % (self.params_dict["core"], \
                self.params_dict["run_nr"], self.params_dict["top_bag"], self.params_dict["bottom_bag"]))

            ##Plotting 2nd derivative with possible loading flags as red bullets
            fig3, ax3 = plt.subplots(ncols=1, nrows=1, num=3, figsize=(10,6), tight_layout=True)
            ax3.plot(self.secs, self.diff2_counts, color="k", linewidth=1)
            ax3.plot(self.secs[np.where(condition_loading)[0]], self.diff2_counts[np.where(condition_loading)[0]], "r.")
            ax3.set_xlabel('Time [s]')
            ax3.set_ylabel('Diff2-counts')
            ax3_1 = ax3.twinx()
            ax3_1.set_ylabel("Sample ON", rotation=270)
            ax3_1.yaxis.set_label_coords(1.06, 0.5)
            ax3_1.plot(self.secs, self.sample_on, "r", linewidth=1)
            ax3.set_title("Core: %s, Run: %s, Bags %s-%s" % (self.params_dict["core"], \
                self.params_dict["run_nr"], self.params_dict["top_bag"], self.params_dict["bottom_bag"]))

        return


    def clip_filter_loading(self, plot_data = True):
        N = np.size(self.diff_counts)
        self.core_loading_indexes = np.genfromtxt("./core_loading_positions.txt", delimiter=",").astype(int)

        # Clipping signal for negative values
        self.diff_counts_clean_1 = copy.deepcopy(self.diff_counts)
        self.diff_counts_clean_2 = np.clip(self.diff_counts_clean_1, a_min = \
            self.params_dict["clipping_min"], a_max=None)

        #Cleaning up for loading sections
        # self.diff_counts_clean_1 = copy.deepcopy(self.diff_counts)
        for j in range(np.shape(self.core_loading_indexes)[0]):
            self.diff_counts_clean_2[self.core_loading_indexes[j,0]:self.core_loading_indexes[j, 1] + 1] = \
                self.params_dict["diff_counts_while_loading"]

        # Convolve with a window
        condition_window = (self.params_dict["filter_size"] > 3) & (type(self.params_dict["filter_size"]) == int)
        if not condition_window:
            print("wrong window size - choose integer>3")
            return

        y_tofilter = np.concatenate((self.diff_counts_clean_2[::-1], self.diff_counts_clean_2, \
            self.diff_counts_clean_2[::-1]))
        window_eval = eval("scipy.signal.windows." + self.params_dict["filter_type"] + "(" + str(self.params_dict["filter_size"]) + ")")
        y_convolved = scipy.signal.convolve(y_tofilter, window_eval, \
            mode="same", method="direct")/np.sum(window_eval)
        self.diff_counts_clean_3 = y_convolved[N: 2 * N]

        if plot_data:
            self.ax2.plot(self.secs, self.diff_counts_clean_1, color="b", linewidth=1, label="loading cleaned")
            self.ax2.plot(self.secs, self.diff_counts_clean_2, color="g", linewidth=1, label="clipped")
            self.ax2.plot(self.secs, self.diff_counts_clean_3, color="r", linewidth=1, label="filtered")
            self.ax2.legend(frameon=False, fontsize=14)

        return


    def integrate_diff_counts(self):
        ##Integration of the 1st derivative signal for total length
        self.length_clipped = np.cumsum(self.diff_counts_clean_2[self.sample_on > 0]) / self.params_dict["counts_per_mm"]
        self.length_filtered = np.cumsum(self.diff_counts_clean_3[self.sample_on > 0]) / self.params_dict["counts_per_mm"]

        return self.length_clipped, self.length_filtered

    def fit_length(self, length_melted):
        solution = optimize.brentq(self.root_function, - 100, 100, args=(length_melted), xtol=1e-6)
        self.ax2.plot(self.secs, self.diff_counts_clean_1, color="b", linewidth=1, label="loading cleaned")
        self.ax2.plot(self.secs, self.diff_counts_clean_2, color="g", linewidth=1, label="clipped")
        self.ax2.plot(self.secs, self.diff_counts_clean_3, color="r", linewidth=1, label="filtered")
        self.ax2.legend(frameon=False, fontsize=14)
        self.params_dict["diff_counts_while_loading"] = solution
        return solution

    def root_function(self, diff_counts_while_loading, length_melted):
        self.params_dict["diff_counts_while_loading"] = diff_counts_while_loading
        self.clip_filter_loading(plot_data=False)
        func_out = self.integrate_diff_counts()[1][-1] - length_melted
        print(func_out)

        return func_out

    def export_encoder_file(self):
        dataout = np.transpose(np.vstack((self.secs[self.sample_on > 0] - self.secs[self.sample_on > 0][0],\
            self.epoch[self.sample_on > 0], self.length_filtered)))

        f_name_out = "%s_run_%s_encoder.out" % (self.params_dict["core"], self.params_dict["run_nr"])
        encoder_out_filepath = tkinter.filedialog.asksaveasfilename(initialdir="../", \
            initialfile = f_name_out, defaultextension=".out", title="Save encoder out file")
        f = open(encoder_out_filepath, "w")
        f.write("#Core, %s\n" % self.params_dict["core"])
        f.write("#Run_nr, %s\n" % self.params_dict["run_nr"])
        f.write("#Top bag, %s\n" % self.params_dict["top_bag"])
        f.write("#Bottom bag, %s\n" % self.params_dict["bottom_bag"])
        f.write("#Loading differential counts used, %0.3f\n" % self.params_dict["diff_counts_while_loading"])
        f.write("#Counts per mm, %0.1f\n" % self.params_dict["counts_per_mm"])
        f.write("#Intervals in secs used for core loading:\n")

        for j in range(np.shape(self.core_loading_indexes)[0]):
            f.write("#%i, %i\n" % (self.core_loading_indexes[j, 0], self.core_loading_indexes[j, 1] + 1))


        f.write("#\nElapsed_time\tEpoch_time\tTotal_length\n")
        np.savetxt(f, dataout, fmt="%0.2f\t%0.2f\t%0.2f")
        f.close()



a = EncoderProcessor()
# a.build_params_json()
a.read_params_json()
print(a.params_dict)
a.read_cfa_file()
solution = a.fit_length(a.params_dict["run_length_melted"])
print(solution)
print(a.length_filtered)
a.export_encoder_file()
