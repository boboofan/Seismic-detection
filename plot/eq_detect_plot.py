import matplotlib.pyplot as plt
import obspy
import csv
import os
import numpy as np

from config.config import Config


def visualise(result_file, max_plot_num=None):
    '''
    :param result_file: csv file 
    :param max_plot_num: 
    :return: 
    '''
    plot_dict = dict()
    with open(result_file) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) == 0:
                continue
            if plot_dict.get(row[0] + '.BHE') == None:
                plot_dict[row[0] + '.BHE'] = [[obspy.UTCDateTime(row[1]), obspy.UTCDateTime(row[2])]]
            else:
                plot_dict[row[0] + '.BHE'].append([obspy.UTCDateTime(row[1]), obspy.UTCDateTime(row[2])])

    config = Config()
    plot_key = list(plot_dict.keys())
    np.random.shuffle(plot_key)
    for file in plot_key:
        file_path = os.path.join(config.re_data_foldername, 'after', file)
        st = obspy.read(file_path)[0]
        plt.plot(st.data, 'k')
        for event in plot_dict[file]:
            event_start = int((event[0] - st.stats.starttime) * 100)
            event_end = int((event[1] - st.stats.starttime) * 100)
            plt.axvline(event_start, color='b')
            plt.axvline(event_end, color='b')
            plt.axvspan(event_start, event_end, ymin=0.1, ymax=0.1, color='r', linestyle='--')

        plt.show()


if __name__ == '__main__':
    visualise(r'C:\Users\linjf\earthquake\code\event_detect\detect_result\cldnn\events_test.csv')
