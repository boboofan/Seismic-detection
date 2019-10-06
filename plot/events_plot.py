import matplotlib.pyplot as plt
import obspy
import csv
import os
import numpy as np

from config.config import Config



def events_plot(result_file):
    '''
    :param result_file: csv file 
    :param max_plot_num: 
    :return: 
    '''
    plot_list = list()
    with open(result_file) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) == 0:
                continue
            plot_list.append(row)

    np.random.shuffle(plot_list)

    config = Config()
    for event in plot_list:
        event_path = os.path.join(config.re_data_foldername, 'after', event[0])
        starttime = obspy.UTCDateTime(event[1])
        endtime = obspy.UTCDateTime(event[2])
        data_e = obspy.read(event_path + '.BHE', starttime=starttime, endtime=endtime)[0].detrend('linear').filter('bandpass', freqmin=4, freqmax=49).data
        data_n = obspy.read(event_path + '.BHN', starttime=starttime, endtime=endtime)[0].detrend('linear').filter('bandpass', freqmin=4, freqmax=49).data
        data_z = obspy.read(event_path + '.BHZ', starttime=starttime, endtime=endtime)[0].detrend('linear').filter('bandpass', freqmin=4, freqmax=49).data

        print(event[-1])

        plt.subplot(3, 1, 1)
        plt.plot(data_e)
        plt.subplot(3, 1, 2)
        plt.plot(data_n)
        plt.subplot(3, 1, 3)
        plt.plot(data_z)
        plt.show()

if __name__ == '__main__':
    events_plot(r'C:\Users\linjf\earthquake\code\event_detect\detect_result\cldnn\events_test.csv')
