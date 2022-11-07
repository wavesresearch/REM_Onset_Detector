import mne
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time as timer
import datetime
import samplerate
from scipy.signal import butter, filtfilt

################  Weighted Movment average Filter #################
def weighted_moving_average(data,sample_rate):
    n=int((7/80)*sample_rate)
    weights = np.arange(1, n + 1)

    filter_data = pd.DataFrame(data)
    wmaN = filter_data.rolling(n).apply(lambda value: np.dot(value, weights) / weights.sum(), raw=True)

    return wmaN.values[:, 0]

# def exp_weighted_moving_average(data,sample_rate):
#     n = int((7 / 80) * sample_rate / 2)
#     filter_data = pd.DataFrame(data)
#     ema10 = filter_data.ewm(span=n).mean()
#     #wmaN = filter_data.rolling(n).apply(lambda value: np.dot(value, weights) / weights.sum(), raw=True)
#
#     return np.array(ema10[0])

# def butter_lowpass_filter(data, cutoff, fs, order):
#     normal_cutoff = cutoff / nyq
#     # Get the filter coefficients
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data)
#     return y

def D(xlist,ylist):
    yprime = np.diff(ylist)/np.diff(xlist)
    xprime = []
    for i in range(len(yprime)):
        xstep = (xlist[i+1]+xlist[i])/2
        xprime = np.append(xprime,xstep)
    return xprime, yprime

def Amplitude_finder(data,sample_rate):
    n=int((30/80)*sample_rate)
    # Looking at 40 (0.5s) to see if value change is greater then 2*30 micro volt as
    # Should change to looking at smaller time then 0.5 sec also !!!!!!
    data_frame = pd.DataFrame(data)
    value_change = data_frame.rolling(n).apply(lambda value: abs(value[0:4].mean()-value[-5:-1].mean()), raw=True)
    index = np.where(abs(value_change) > 2*30/10**6)
    index_fixed = np.array([x - int(n) for x in index[0]])
    return index_fixed

def remove_lonly_elements(arr,sample_rate):
    n=int((5/80)*sample_rate)
    # Removes elements with less then n elements close to it
    # Not in use anymore
    index_set = set()
    for i in range(len(arr) - n):
        if arr[i + n] - arr[i] <= n + 1:
            for value in arr[i:i + n + 1]:
                # print(arr[i:i+n])
                index_set.add(value)
    index_list = list(index_set)
    index_list.sort()
    return index_list

def ok_slope_average(arr,sample_rate):
    n=int((10/80)*sample_rate)
    # Check for where mean of next n values are over criteria and where the fist value is over the criteria
    criteria_value = 2 * 250 / 10 ** 6
    data_frame = pd.DataFrame(arr)
    # find the mean of the slope over n consecutive values
    value_change = data_frame.rolling(n).apply(lambda value: np.mean(value), raw=True)
    # move the array n spots so pos 0 = mean for 0 - n
    arr_mean_value = np.array(value_change[n-1:-1][0])
    arr_out = []
    # find indexes for where the value are greater then criteria
    index_value_over = np.where(abs(arr) > criteria_value)
    # check for mean over next n points is over criteria and if start point is over criteria
    for i in range(len(arr_mean_value)):
        if abs(arr_mean_value[i]) > criteria_value :
            if i in index_value_over[0]:
                arr_out = np.append(arr_out,i)
    # return indexes of points that meet both criterias
    return np.array(arr_out,int)

def Slope_finder(data_prime,sample_rate):

    # Finding indexes of points where data dirivative is greater then 2*248.3 micro volts
    #points_value = data_prime[np.where(data_prime > 2*248.3/10**6)]
    #point_index = np.where(abs(data_prime) > 2*248.3/10**6 )

    # Only accept where its n point in a row
    #indexes = remove_lonly_elements(point_index[0],sample_rate)
    # Only accept when average slope over 10 points is over 2*248.3
    indexes = ok_slope_average(data_prime,sample_rate)
    return indexes

def find_index_2prime(EMs_points,y2prime,data,sample_rate):
    n = int((5 / 80) * sample_rate)
    # Finding maximum of double derivative function close to the points
    # This should give the start of the peak
    # Make sure the maximum is in the right direction (if value goes down it needs to be the minimum of the y2prime)
    indexes_EMs = []
    for value in EMs_points:
        value = int(value)
        if value <= n:
            indexes_EMs.append(value)
            continue
        if data[value] < data[value+2*n]:
            index = np.argmax(y2prime[value - n:value+n]) # value
        else:
            index = np.argmin(y2prime[value - n:value+n]) # value
        indexes_EMs.append(value-n+index)

    indexes_EMs = np.array(indexes_EMs)
    return indexes_EMs

def remove_close_elements(arr,sample_rate):
    # Remove elements to close to approved element. to close is definded by closness
    # there needs to be a gap bigger then closeness between point a and b for the point b to be accepted
    remove_after = set()
    closeness = 0.2 * sample_rate # 0.2 sec
    for i in range(len(arr)-1):
        if abs(arr[i+1] - arr[i]) < closeness:
            remove_after.add(i+1)
    if len(remove_after) < 1:
        return arr
    new_arr = np.delete(arr, np.array(list(remove_after)))
    return new_arr.astype(int)

def find_EMs(data_double_prime,slopes,amplitudes,data,sample_rate):
    # Finding indexes where both the slope and amplitude is criteria is meet
    shared_values = []
    for value in slopes:
        if value in amplitudes:
            shared_values = np.append(shared_values, value)
    if len(shared_values) < 1:
         return []

    return_array = remove_close_elements(shared_values,sample_rate)
    # if len(return_array) < 1:
    #     return []
    # move point to where the double derivative is larges, should be at the start of a slope
    hold = find_index_2prime(return_array,data_double_prime,data,sample_rate)
    hold = set(hold)
    hold = np.array(list(hold))
    return hold

def check_opposite_spike(EM_points,EOGl,EOGr,sample_rate):
    n = int(0.2*sample_rate)
    criteria = 30/10**6 # 30 til 45 til 55
    point_ok = []
    for value in EM_points:
        left_spike = np.mean(EOGl[value+n-int(n/20):value+n+int(n/20)]) - np.mean(EOGl[value-int(n/20):value+int(n/20)])
        right_spike = np.mean(EOGr[value+n-int(n/20):value+n+int(n/20)]) - np.mean(EOGr[value-int(n/20):value+int(n/20)])
        if (left_spike > criteria and right_spike < criteria) or (left_spike <criteria and right_spike > criteria):
            point_ok = np.append(point_ok,value)
    return np.array(point_ok,int)

def check_induvidual_channels(EM_points,EOG_left,EOG_right,sample_rate):
    step = int(0.08*sample_rate) # = 50 milliseconds
    n = 4 # for å ungå spike points
    first_step = int(0.021*sample_rate)
    criteria = 14/10**6
    point_ok = []
    for em in EM_points:
        right_em_spike = 0
        left_em_spike = 0
        left_spike = 0
        right_spike = 0
        for i in range(1,4):
            if i == 1:
                left_spike = np.mean(EOG_left[em - n + step:em + n + step]) - np.mean(EOG_left[em - n - first_step:em + n + -first_step])
                right_spike = np.mean(EOG_right[em - n + step:em + n + step]) - np.mean(EOG_right[em - n - first_step:em + n + -first_step])
            # else:
            #     left_spike = np.mean(EOG_left[em-n+step*(1+i):em+n+step*(1+i)]) -np.mean(EOG_left[em-n+step*i:em+n+step*i])
            #     right_spike = np.mean(EOG_right[em-n+step*(1+i):em+n+step*(1+i)]) - np.mean(EOG_right[em-n+step*i:em+n+step*i])
            if (left_spike > criteria and right_spike < -criteria):
                right_em_spike += 1
            if (left_spike < -criteria and right_spike > criteria):
                left_em_spike += 1
        if left_em_spike >= 1 or right_em_spike >= 1:
            point_ok = np.append(point_ok,em)
    return np.array(point_ok,int)

class REM_Detector:

    def __init__(self, data, file_name, folder_name, EOGs_channels, EMG_c_channel_name, sample_rate=250, verbose=False):

        self.file_name = file_name
        self.folder_name = folder_name
        self.data = data

        event_name = '{}-EMs-eve.fif'.format(self.file_name)
        self.event_new_fname = self.folder_name + '/' + event_name

        self.verbose = verbose
        self.events_found = []
        self.sample_rate = sample_rate
        self.EOGs_channels = EOGs_channels
        self.EMG_c_channel_name = EMG_c_channel_name

        self.input_sfreq = self.data.info['sfreq']

        #################### LOW PASS FILTER ############################
        # Filter requirements.
        self.fs = sample_rate  # sample rate, Hz
        self.cutoff = 200  # desired cutoff frequency of the filter, Hz ,slightly higher than actual 1.2 Hz
        self.nyq = 0.5 * self.fs  # Nyquist Frequency
        self.order = 8  # sin wave can be approx represented as quadratic

    def Detecting_EMs(self):
        tmin, tmax = 0, len(self.data) / self.input_sfreq - 0.1
        time_per_loop = 100
        nr_of_runs_needed = (tmax - tmin) / time_per_loop
        actual_max = tmax
        actual_min = tmin
        EM_points_saved = []

        sample_rate = self.sample_rate
        sample_ratio = self.sample_rate / self.input_sfreq
        converter = 'sinc_best'  # or 'sinc_fastest', ...

        start_time = timer.time()

        print('Running algo on tmin = {}, tmax = {} , number of runs = {}'.format(actual_min, actual_max,
                                                                          nr_of_runs_needed))
        for long_run in range(int(nr_of_runs_needed) + 1):
            loop_start = timer.time()
            if long_run > int(nr_of_runs_needed) - 1:
                tmin = actual_min + long_run * time_per_loop
                tmax = actual_max
            else:
                tmin = actual_min + long_run * time_per_loop
                tmax = tmin + time_per_loop

            raw = self.data.copy()
            raw = raw.crop(tmin=tmin, tmax=tmax)
            raw.load_data(verbose='WARNING')
            # alt må inn her!

            # raw.notch_filter(50)

            sfreq = self.data.info['sfreq']
            time = raw.pick_channels(self.EOGs_channels).times
            data = raw.pick_channels(self.EOGs_channels)._data
            scalar = 10 ** 6

            EOG_left = data[0]
            EOG_right = data[1]
            # data = np.subtract(weighted_moving_average(data[1]),weighted_moving_average(data[0]))
            data_samlet = np.subtract(EOG_right, EOG_left)

            data_sampled = samplerate.resample(data_samlet, sample_ratio, converter)
            time_sampled = samplerate.resample(time, sample_ratio, converter)

            time = time_sampled

            data = weighted_moving_average(data_sampled,sample_rate)

            EOG_left = samplerate.resample(EOG_left, sample_ratio, converter)
            EOG_right = samplerate.resample(EOG_right, sample_ratio, converter)

            ################### Derivasjon ##########################
            xprime, yprime = D(time, data)
            x2prime, y2prime = D(xprime, yprime)
            amplitude_marker = Amplitude_finder(data,sample_rate)
            derivert_marker = Slope_finder(yprime, sample_rate)
            EM_points = find_EMs(y2prime, derivert_marker, amplitude_marker, data, sample_rate)
            EM_points_checked2 = check_induvidual_channels(EM_points, EOG_left, EOG_right, sample_rate)

            ################## Make EM points events for data ##################################
            new_events = np.zeros(shape=(len(EM_points_checked2), 3), dtype=int)

            event_time = [time[x] for x in EM_points_checked2]
            event_time.sort()
            event_time = [x + tmin for x in event_time]
            event_time_new = np.array([int(x * sample_rate / sample_ratio) for x in event_time])

            new_events[:, 0] = event_time_new
            new_events[:, 2] = np.ones(len(event_time_new))

            EM_points_saved.append(new_events)

            loop_end = timer.time()
            total_seconds_in_loop = loop_end - start_time
            loop_seconds = loop_end - loop_start
            loop_time = str(datetime.timedelta(seconds=loop_seconds))
            total_time_in_loop = str(datetime.timedelta(seconds=total_seconds_in_loop))
            # print('Total time of loop {}/{} = {}, total = {}'.format(long_run+1, int(nr_of_runs_needed+1), loop_time,
            #                                                          total_time_in_loop))
            print('Loop nr {}/{}, tmin = {}, tmax = {}'.format(long_run+1, int(nr_of_runs_needed+1), tmin, tmax))
            print('Nr of events in this loop = {}'.format(len(EM_points_checked2)))

        end_time = timer.time()
        total_seconds = end_time - start_time

        total_time = str(datetime.timedelta(seconds=total_seconds))

        print('Total time of the script = {}'.format(total_time))

        ########################## Reformat events to standard event format ######################################
        total_event_lenght = 0
        for arr in EM_points_saved:
            total_event_lenght += len(arr)
        self.events_found = np.zeros(shape=(total_event_lenght, 3), dtype=int)

        length_to_now = 0
        for i, arr in enumerate(EM_points_saved):
            self.events_found[length_to_now:length_to_now + len(EM_points_saved[i])] = arr
            length_to_now += len(EM_points_saved[i])
        print('nr of events found = {}'.format(len(self.events_found[:, 0])))
        events_per_min = len(self.events_found[:, 0]) * 60 / actual_max
        print('Events per minute = {}'.format(events_per_min))


    def PSD_Checking_EMs(self,PSD_threshold=-10):
        tmin, tmax = 0, len(self.data) / self.input_sfreq - 0.1
        event = self.events_found
        start_stop = []
        time_span = 5
        for i, time in enumerate(event[:, 0]):
            # find tmin and tmax
            if time < time_span * self.input_sfreq or int(time / self.input_sfreq) > tmax - time_span:
                start_stop.append([0, 0])
                continue
            start_stop.append(
                [int(time / self.input_sfreq - time_span), int(time / self.input_sfreq + time_span)])

        meanvalue = []
        print('Checking PSD of {} EM '.format(len(start_stop)))
        for i, time in enumerate(start_stop):  # start_stop
            if time[0] < 10:
                time[1] = time[0] + 10
            # if i < 1000 or i > 1500:
            #     continue
            raw = self.data.copy()
            raw.pick(self.EMG_c_channel_name)  # 'EMG-c'
            raw.crop(tmin=time[0], tmax=time[1])
            raw.load_data(verbose='WARNING')
            fmin, fmax = 55, 95
            pas, freqs = mne.time_frequency.psd_array_welch(raw._data, self.input_sfreq, fmin=fmin, fmax=fmax,
                                                            verbose='WARNING')

            pas[0] = pas[0] * 10 ** 12
            pas[0] = 10 * np.log10(pas[0])  # This to convert to dB
            # plt.plot(freqs,pas[0],'--')
            meanvalue.append(sum(pas[0]) / len(pas[0]))
            # print('Processing EM {} / {}'.format(i + 1, len(start_stop)))

        new_EMs = []
        for i, value in enumerate(meanvalue):
            if value < PSD_threshold:
                new_EMs.append(event[i, 0])

        events_found = np.zeros(shape=(len(new_EMs), 3), dtype=int)

        events_found[:, 0] = new_EMs
        events_found[:, 2] = np.ones(len(new_EMs), dtype=int)
        self.events_found = events_found

    def Saving_EMs(self):
        mne.write_events(self.event_new_fname,self.events_found, overwrite=True)

    def Reading_EMs(self):
        self.events_found = mne.read_events(self.event_new_fname)

    def Export_EMs_csv(self):
        self.events_found = mne.read_events(self.event_new_fname)
        events2 = self.events_found[:,0] / self.input_sfreq
        event_name = '{}-EMs-eve.csv'.format(self.file_name)
        data_name = self.folder_name+ '/' + event_name
        pd.DataFrame(events2).to_csv(data_name, sep=';')
        print('File "' + data_name + '" Generated')

    def Plotting_EMs(self):
        test = self.data.copy().pick_channels(self.EOGs_channels)
        test.plot(events=self.events_found)
        plt.show()

    def Export_EMs_edf(self):
        ######## Convert to EDF channels EOG and Events
        EOGs = self.data.copy().pick_channels(self.EOGs_channels)

        # Get EOGs channels data
        EOGs_data = EOGs.get_data()
        events = self.events_found

        # Convert events data in a channel data structure
        Events_data = np.zeros((1, EOGs_data.shape[1]))
        for i in events[:, 0]:
            Events_data[0, i] = 1

        # Join EOG and Event Channels
        Data = np.append(EOGs_data, Events_data, axis=0)

        # Create a Simulated Raw
        ch_names = [self.EOGs_channels[0],self.EOGs_channels[1], 'EMs']
        ch_types = ['eeg', 'eeg', 'eeg']
        info = mne.create_info(ch_names, ch_types=ch_types, sfreq=self.input_sfreq, verbose='WARNING')
        simulated_raw = mne.io.RawArray(Data, info)

        event_name = '{}-EMs-eve.edf'.format(self.file_name)
        data_name = self.folder_name+ '/' + event_name
        mne.export.export_raw(data_name, simulated_raw, fmt='auto', add_ch_type=True, overwrite=True, verbose='WARNING')
        print('File "' + data_name + '" Generated')


def main():

    file_name = 'T5-1.edf'  # T15-2  # Adjust according to the file name
    folder_name = 'LIM_T5-1'  # Adjust according to the file folder

    EOG_l_channel_name = 'EOG L-A2'   # 'EOG-l'
    EOG_r_channel_name = 'EOG R-A2'  # 'EOG-r'
    EMG_c_channel_name = 'EMG1-EMG2'
    EOGs_channels = [EOG_l_channel_name, EOG_r_channel_name]

    data_path = os.path.expanduser(folder_name)
    raw_fname = os.path.join(data_path, file_name)
    raw = mne.io.read_raw_edf(raw_fname, preload=False)
    raw = raw.crop(tmin=0, tmax=400)

    Detector = REM_Detector(raw, file_name, folder_name, EOGs_channels, EMG_c_channel_name, sample_rate=250)
    Detector.Detecting_EMs()
    # Detector.PSD_Checking_EMs()

    Detector.Saving_EMs()
    Detector.Reading_EMs()
    Detector.Export_EMs_edf()
    Detector.Plotting_EMs()
    Detector.Export_EMs_csv()

if __name__ == "__main__":
    main()
