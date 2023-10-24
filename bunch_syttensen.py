"""
#######################################################################################
Changes history for bunch_syttensen.py
#
#03112014: Added pickle_bunch and unpickle_bunch methods.
#          In this version it is the dictionary __dict__ of the instance that
#          is pickles instead of the whole instance. IN this way back compatibility
#          issues are less likely to happen. unpickle_bunch method loads
#          the dictionary creates a bunch instance and assigns all the keys of the
#          loaded dictionary to the keys of the __dict__ of the instance.
#03112014: Method interp_depth is added in the Run class. Interpolates self.z on a scale
#          defined by coordinates with time and known z markers.
#14112015: Changed the order of params in the smow bunch method p_smow_d17 is now first
#          Added Run method for reading the depth registration data from recap files
#          Changed the save_ascii method such that self.z is in meters
#
#20171026: Corrected small error in method read data. In the self.baseline block it was
#          by mistake the self.p_cav that was filled with -999 in case there was an error in reading
#          and not the self.baseline array
#
#20190123: Added prune_on_depth method. Similar to the existing prune method performing
#          the pruning using depth sections
#
#######################################################################################
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import os.path
import time
import copy
import pickle
import json



class Bunch(object):

    def __init__(self):
        """
        def __init__(self):
        self.index_i = None
        self.date_i = None
        self.time_i = None
        self.days_since_jan = None
        self.temp_cav = None
        self.p_cav = None
        self.time_delay = None
        self.baseline = None
        self.h2o = None
        self.d18 = None
        self.dD = None
        """


        self.index_i = np.array(())
        self.date_i = np.array(())
        self.time_i = np.array(())
        self.days_since_jan = np.array(())
        self.temp_cav = np.array(())
        self.p_cav = np.array(())
        self.time_delay = np.array(())
        self.secs = np.array(())
        self.baseline = np.array(())
        self.h2o = np.array(())
        self.d18 = np.array(())
        self.dD = np.array(())
        self.z = np.array(())
        self.epoch = np.array(())
        self.d17 = np.array(())
        self.peak1 = np.array(())
        self.peak2 = np.array(())
        self.peak3 = np.array(())
        self.peak11 = np.array(())
        self.peak13 = np.array(())
        self.peak1_offset = np.array(())
        self.peak2_offset = np.array(())
        self.peak3_offset = np.array(())
        self.D17 = np.array(())

        return

    def read_data(self, filename):
        """

        """

        data = np.loadtxt(filename, dtype = "S")
        header = data[0,:]
        data = data[1:,:]


        try:
            index_date = np.where(header == b"DATE")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_days_since_jan = np.where(header == b"FRAC_DAYS_SINCE_JAN1")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_epoch = np.where(header == b"EPOCH_TIME")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_water_ppm = np.where(header == b"H2O")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_peak1 = np.where(header == b"peak1")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_peak2 = np.where(header == b"peak2")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_peak3 = np.where(header == b"peak3")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_peak11 = np.where(header == b"peak_11")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_peak13 = np.where(header == b"peak_13")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_peak1_offset = np.where(header == b"peak1_offset")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_peak2_offset = np.where(header == b"peak2_offset")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_peak3_offset = np.where(header == b"peak3_offset")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_d18 = np.where(header == b"Delta_18_16")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_dD = np.where(header == b"Delta_D_H")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_d17 = np.where(header == b"Delta_17_16")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_D17 = np.where(header == "Excess_17")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_cavity_temp = np.where(header == b"CavityTemp")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_cavity_pressure = np.where(header == b"CavityPressure")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            index_baseline_shift = np.where(header == b"baseline_shift")[0][0]
        except:
            print("Some of the data fields were not read properly")

        try:
            self.epoch = data[:,index_epoch].astype("float")
            self.date_i = data[:,index_date].astype("str")
            self.days_since_jan = data[:,index_days_since_jan].astype("float")
            time_diff = np.diff(self.epoch)
            self.time_delay = np.append(time_diff, np.mean(time_diff))
            self.index_i = np.arange(np.size(self.time_delay))
            mean_time_delay = np.mean(self.time_delay)
            # self.secs = np.cumsum(self.time_delay)
            self.secs = self.epoch - self.epoch[0]
            print("new secs")
        except:
            print("time not read")


        try:
            self.h2o = data[:,index_water_ppm].astype("float")
        except:
            print("Water_ppm not read")
            self.h2o = 20000. + np.zeros(np.size(self.epoch))


        try:
            self.peak1 = data[:,index_peak1].astype("float")
        except:
            print("peak1 not read")
            self.peak1 = 999. + np.zeros(np.size(self.epoch))

        try:
            self.peak2 = data[:,index_peak2].astype("float")
        except:
            print("peak2 not read")
            self.peak2 = 999. + np.zeros(np.size(self.epoch))

        try:
            self.peak3 = data[:,index_peak3].astype("float")
        except:
            print("peak3 not read")
            self.peak3 = 999. + np.zeros(np.size(self.epoch))

        try:
            self.peak11 = data[:,index_peak11].astype("float")
        except:
            print("peak11 not read")
            self.peak11 = 999. + np.zeros(np.size(self.epoch))

        try:
            self.peak13 = data[:,index_peak13].astype("float")
        except:
            print("peak13 not read")
            self.peak13 = 999. + np.zeros(np.size(self.epoch))

        try:
            self.peak1_offset = data[:,index_peak1_offset].astype("float")
        except:
            print("peak1_offset not read")
            self.peak1_offset = 999. + np.zeros(np.size(self.epoch))


        try:
            self.peak2_offset = data[:,index_peak2_offset].astype("float")
        except:
            print("peak2_offset not read")
            self.peak2_offset = 999. + np.zeros(np.size(self.epoch))


        try:
            self.peak3_offset = data[:,index_peak3_offset].astype("float")

        except:
            print("peak3_offset not read")
            self.peak3_offset = 999. + np.zeros(np.size(self.epoch))


        try:
            self.d18 = data[:, index_d18].astype("float")
        except:
            print("d18 not read")
            self.d18 = 999. + np.zeros(np.size(self.epoch))


        try:
            self.dD = data[:, index_dD].astype("float")
        except:
            print("dD not read")
            self.dD = 999. + np.zeros(np.size(self.epoch))

        try:
            self.d17 = data[:, index_d17].astype("float")
        except:
            print("d17 not read")
            self.d17 = 999. + np.zeros(np.size(self.epoch))


        try:
            self.D17 = data[:, index_D17].astype("float")
        except:
            print("D17 not calculated")
            self.D17 = 999. + np.zeros(np.size(self.epoch))


        try:
            self.temp_cav = data[:, index_cavity_temp].astype("float")
        except:
            print("cavity temp not calculated")
            self.temp_cav = 999. + np.zeros(np.size(self.epoch))


        try:
            self.p_cav = data[:, index_cavity_pressure].astype("float")
        except:
            print("p_cav not calculated")
            self.p_cav = 999. + np.zeros(np.size(self.epoch))

        try:
            self.baseline = data[:, index_baseline_shift].astype("float")
        except:
            print("baseline not calculated")
            self.baseline = 999. + np.zeros(np.size(self.epoch))

        for i in self.__dict__:
            print((i, np.shape(self.__dict__[i])))

        return



    def concat(self, bunch_1):
        """
        appends all arrays of instance with the arrays of bunch_1
        sorts all indexes based on the epoch time of the
        final bunch_1 epoch array
        """
        for j in list(self.__dict__.keys()):
            self.__dict__[j] = np.hstack((self.__dict__[j],
                bunch_1.__dict__[j]))
        sort_at = np.argsort(self.epoch)
        ignore_list = ["time_i",]

        for j in list(self.__dict__.keys()):
            print(j)
            if j in ignore_list:
                continue
            try:
                self.__dict__[j] = self.__dict__[j][sort_at]
            except:
                print(("%s was not sorted properly" %j))
                continue
        self.secs = self.epoch - self.epoch[0]
        self.index_i = np.arange(np.size(self.epoch))

        return



    def pickle_bunch(self, filename):
        """
        Pickles the __dict__ of the bunch instance in a file
        """
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f)
        f.close()

        return



    def unpickle_bunch(self, filename):
        """

        """
        f = open(filename, 'rb')
        unpickled_dict = pickle.load(f)
        f.close()

        for i in list(unpickled_dict.keys()):
            if i not in list(self.__dict__.keys()):
                print(("Error loading key %s to bunch instance" %i))
            else:
                for j in list(self.__dict__.keys()):
                    if i == j:
                        self.__dict__[j] = unpickled_dict[i]
                    else:
                        continue


        return


    def save_mat(self, filename):
        """

        """
        sp.io.savemat(filename, self.__dict__)
        return


    def save_ascii(self, filename, z = False, verbose = False, reset_index = False):
        """

        """
        if reset_index:
            self.index_i = np.arange(np.size(self.dD))

        criterion0 = (verbose == False) & (z == False)
        criterion1 = (verbose == False) & (z == True)
        criterion2 = (verbose == True) & (z == False)
        criterion3 = (verbose == True) & (z == True)
        if criterion0:
            f = open(filename, "w")
            f.write("Index\ttime_epoch\td18\tdD\td17\tD17excess\n")
            formats = ("%0.0f\t%0.3f\t%0.5f\t%0.5f\t%0.5f\t%0.8f")
            data_out = np.transpose(np.vstack\
                ((self.index_i, self.epoch, self.d18, self.dD, self.d17, self.D17)))
            np.savetxt(f, data_out, delimiter = "\t", fmt = formats)
            f.close()

        elif criterion1:
            f = open(filename, "w")
            f.write("Index\ttime_epoch\tDepth\td18\tdD\td17\tD17excess\n")
            formats = ("%0.0f\t%0.3f\t%0.5f\t%0.5f\t%0.5f\t%0.5f\t%0.8f")
            data_out = np.transpose(np.vstack\
                ((self.index_i, self.epoch, self.z, self.d18, self.dD, self.d17, self.D17)))
            np.savetxt(f, data_out, delimiter = "\t", fmt = formats)
            f.close()

        elif criterion2:
            f = open(filename, "w")
            f.write("Index\ttime_epoch\td18\tdD\td17\tD17excess\th2o\ttemp_cavity\tpress_cavity\n")
            formats = ("%0.0f\t%0.3f\t%0.5f\t%0.5f\t%0.5f\t%0.8f\t%0.4f\t%0.5f\t%0.5f")
            data_out = np.transpose(np.vstack\
                ((self.index_i, self.epoch, self.d18, self.dD, self.d17, self.D17, self.h2o, self.temp_cav, self.p_cav)))
            np.savetxt(f, data_out, delimiter = "\t", fmt = formats)
            f.close()

        elif criterion3:
            f = open(filename, "w")
            f.write("Index\ttime_epoch\tDepth\td18\tdD\td17\tD17excess\th2o\ttemp_cavity\tpress_cavity\n")
            formats = ("%0.0f\t%0.3f\t%0.5f\t%0.5f\t%0.5f\t%0.5f\t%0.8f\t%0.4f\t%0.5f\t%0.5f")
            data_out = np.transpose(np.vstack\
                ((self.index_i, self.epoch, self.z, self.d18, self.dD, self.d17, self.D17, self.h2o, self.temp_cav, self.p_cav)))
            np.savetxt(f, data_out, delimiter = "\t", fmt = formats)
            f.close()

        return

    def plot(self):
        """
        def plot(self):
        """

        plt.figure(23)
        plt.clf()

        plt.subplot(311)
        start_time = time.strftime("%a %d%m%y-%H:%M:%S UTC", time.gmtime(self.epoch[0]))
        plt.title("Start: %s" %start_time)
        plt.ylabel("d18")
        plt.plot(self.secs/3600, self.d18, 'b')
        plt.ylim((np.min(self.d18) - 2, np.max(self.d18)+2))
        plt.subplot(312)
        plt.ylabel("dD")
        plt.plot(self.index_i, self.dD, 'r')
        plt.ylim((np.min(self.dD)-5, np.max(self.dD)+5))
        plt.subplot(313)
        plt.ylabel("H2O [ppm]")
        plt.xlabel("Hours")
        plt.plot(self.secs/3600, self.h2o, 'm')

        return


    def lin_trans(self, y, p):
        """

        """
        return np.polyval(p,y)


    def smow(self, p_smow_d17, p_smow_d18, p_smow_dD):
        """

        """
        smow_instance = copy.deepcopy(self)
        smow_instance.dD = self.dD*p_smow_dD[0] + p_smow_dD[1]
        smow_instance.d18 = self.d18*p_smow_d18[0] + p_smow_d18[1]
        #smow_instance.D17 = self.D17*p_smow_D17[0] + p_smow_D17[1]
        smow_instance.d17 = self.d17*p_smow_d17[0] + p_smow_d17[1]
        smow_instance.D17 = np.log(smow_instance.d17/1000+1) - 0.5281*np.log(smow_instance.d18/1000+1)
        return smow_instance


    def humidity(self, slope_d18=1.94, slope_dD=3.77, h2o_ref=20000):
        """

        """
        humidity_instance = copy.deepcopy(self)
        R = humidity_instance.h2o/h2o_ref
        humidity_instance.d18-=slope_d18*(R-1)
        humidity_instance.dD -= slope_dD*(R-1)
        humidity_instance.h2o = humidity_instance.h2o/R

        return humidity_instance


    def pick(self, index_i, index_f):
        """

        """
        new_instance = copy.deepcopy(self)
        for attrib in new_instance.__dict__:
            try:
                expression = "new_instance." + attrib + " = new_instance." + attrib + \
                    "[" + str(index_i) + ":" + str(index_f) + "]"
                exec(expression)
            except TypeError:
                expression = "new_instance." + attrib + " = new_instance." + attrib
                exec(expression)
        return new_instance


    def prune_on_depth(self, z_1 = None, z_2 = None, readfile = None, reset_index = True):
        """
        prunes the self.__dict__ according to indexes or a tab delimited txt file
        """
        if np.size(self.z) == 0:
            print("Depth array is empty. Cannot prune by depth variable.")
            return

        if readfile == None:
            criterion = (self.z > z_1) & (self.z < z_2)
            if np.all(criterion == False):
                print("Nothing to prune..")
                return
            for key in list(self.__dict__.keys()):
                self.__dict__[key] = np.delete(self.__dict__[key], np.where(criterion))
        else:
            f = open(readfile + ".pru", "w")
            pruneat = np.genfromtxt(readfile, delimiter = ",")
            rows = np.shape(pruneat)[0]


            for i in np.arange(rows):
                z_1 = np.float(pruneat[i,0])*1000
                z_2 = np.float(pruneat[i,1])*1000

                criterion = (self.z > z_1) & (self.z < z_2)
                if np.all(criterion == False):
                    print("Nothing to prune..")
                    continue
                str_out = str(z_1) + "\t" + str(z_2)+ "\n"
                print(("Prunning section at depth: %0.4f - %0.4f" %(z_1, z_2)))
                f.write(str_out)

                for key in list(self.__dict__.keys()):
                    try:
                        self.__dict__[key] = np.delete(self.__dict__[key], np.where(criterion))
                    except IndexError:
                        print("Key %s not clipped - likely empty array" %key)
            f.close()

            if reset_index:
                self.index_i = np.arange(np.size(self.z))

        return



    def prune(self, index_i_1 = None, index_i_2 = None, readfile = None):
        """
        prunes the self.__dict__ according to indexes or a tab delimited txt file
        """
        if readfile == None:
            criterion = (self.index_i > index_i_1) & (self.index_i < index_i_2)
            if np.all(criterion == False):
                print("Nothing to prune..")
                return

            for key in list(self.__dict__.keys()):
                self.__dict__[key] = np.delete(self.__dict__[key], np.where(criterion))
        else:
            f = open(readfile + ".pru", "w")
            pruneat = np.loadtxt(readfile)
            rows = np.shape(pruneat)[0]

            for i in np.arange(rows):
                index_i_1 = pruneat[i,0]
                index_i_2 = pruneat[i,1]
                criterion = (self.index_i > index_i_1) & (self.index_i < index_i_2)

                if np.all(criterion == False):
                    print("Nothing to prune..")
                    continue

                z_prune_top = self.z[np.where(criterion)][0]
                z_prune_bottom = self.z[np.where(criterion)][-1]

                str_out = str(self.index_i[np.where(criterion)][0]) + "\t"+ \
                    str(self.index_i[np.where(criterion)][-1]) + "\t" + \
                        str(z_prune_top) + "\t" + str(z_prune_bottom )+ "\n"

                print(("Prunning at: %0.3f - %0.3f" %(z_prune_top, z_prune_bottom)))
                f.write(str_out)

                for key in list(self.__dict__.keys()):
                    self.__dict__[key] = np.delete(self.__dict__[key], np.where(criterion))

            f.close()

        return



    def smooth(self, x,window_len=11,window='hanning'):
        """
        smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal
        """

        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")


        if window_len<3:
            return x


        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


        s=np.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len-1:-window_len+1]






class Run(Bunch):

    def __init__(self, bunch_instance):
        """

        """
        self.__dict__ = copy.deepcopy(bunch_instance.__dict__)
        self.index_i = np.arange(np.size(self.dD))
        self.secs = np.cumsum(self.time_delay) - self.time_delay[0]

        return


    def locate(self, index_i = None, index_f = None):
        """

        """
        smoothed_dD = self.smooth(self.dD, 10, window = "bartlett")
        diff_dD = np.diff(smoothed_dD)
        if index_i == None:
            if index_f == None:
                index_i = np.argmin(diff_dD)
                index_f = np.argmax(diff_dD)
            else:
                pass
        print(index_i, index_f)

        plt.figure(21)
        plt.clf()
        plt.subplot(211)
        plt.plot(self.index_i, smoothed_dD, "b")
        plt.plot([index_i, index_f], [self.dD[index_i], self.dD[index_f]], "ro")
        plt.subplot(212)
        plt.plot(self.index_i[:-1], diff_dD, "b")
        plt.plot([index_i, index_f], [diff_dD[index_i], diff_dD[index_f]], "ro")

        self.__dict__ = copy.deepcopy(self.pick(index_i, index_f)).__dict__
        self.index_i = np.arange(np.size(self.dD))
        self.secs = np.cumsum(self.time_delay) - self.time_delay[0]

        return

    def add_break(self, position = 0, width = 0.2):
        """
        Adds a break of width = width at position = position
        All values in cm..!
        """
        if self.z == None:
            print("z attribute not defined yet for this Run instance")
            return
        else:
            pass

        self.z[self.z>=position]+=width

        print("Remember all values in cm...")

        return


    def interp_depth(self, t_array, depth_array):
        """
        Interpolates the self.z on a scale defined by coordinates t_array (time) and
        depth_array which contains known depth markers during melting.
        """

        t_elapsed = t_array - t_array[0]
        if t_elapsed[-1] - self.secs[-1] > 20:
            raise ValueError("Total time of cfa 20 seconds or longer than the total time of run instance. Inspect carefully..!")
        self.z = np.interp(self.secs, t_elapsed, depth_array)
        return

    def read_recap_depth(self, filename):
        """
        Reads recap depth registration files returns secs, length, true_depth three arrays
        """
        data_depth = np.genfromtxt(filename, skip_header = 1, delimiter = ",",\
            usecols = (0,1,2,3), filling_values = -333.)

        indexes_melt = np.where(data_depth[:,2]!= -333.)[0]
        melt_secs = data_depth[indexes_melt, 0]
        melt_secs = melt_secs - melt_secs[0]
        melt_epoch = data_depth[indexes_melt, 1]
        melt_length = data_depth[indexes_melt, 2]
        melt_depth = data_depth[indexes_melt, 3]

        return melt_secs, melt_epoch, melt_length, melt_depth
