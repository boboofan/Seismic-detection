import csv
import os

import numpy as np
import obspy
from obspy.signal.filter import bandpass
import matplotlib.pyplot as plt
import datetime

from config.config import Config
from reader.reader import Reader


def data_preprocess(data, filter_model='bandpass', isabs=True):
    if filter_model == 'bandpass':
        data = bandpass(data, freqmin=4, freqmax=49, df=100)

    data = (data - np.mean(data)) / (np.max(np.absolute(data)) + 1)
    if isabs:
        data = np.absolute(data)
    return data


def main(model_name, new_scan=False, preprocess=True):
    reader = Reader()
    config = Config()

    reader.aftername.sort()

    if new_scan == True:
        print('start new scan!')
        file_list = reader.aftername

        start_point = 0
    else:
        with open('detect_result/' + model_name + '/checkpoint') as file:
            start_point = int(file.readline())
            file_list = reader.aftername[start_point:]
            print('restart from {}'.format(file_list[0]))

    if model_name == 'cnn':
        from event_detect.cnn import CNN
        import tensorflow as tf
        from tflib.models import Model

        model = CNN()
        sess = tf.Session()
        saver, global_step = Model.continue_previous_session(sess,
                                                             model_file='cnn',
                                                             ckpt_file='saver/cnn/checkpoint')

    if model_name == 'cldnn':
        from event_detect.cldnn import CLDNN
        import tensorflow as tf
        from tflib.models import Model

        model = CLDNN()
        sess = tf.Session()
        saver, global_step = Model.continue_previous_session(sess,
                                                             model_file='cldnn',
                                                             ckpt_file='saver/cldnn/checkpoint')

    for file in file_list:
        begin = datetime.datetime.now()
        traces = obspy.read(file[0])
        traces = traces + obspy.read(file[1])
        traces = traces + obspy.read(file[2])

        if not (traces[0].stats.starttime == traces[1].stats.starttime
                and traces[0].stats.starttime == traces[2].stats.starttime):
            starttime = max([traces[0].stats.starttime,
                             traces[1].stats.starttime,
                             traces[2].stats.starttime])
            for j in range(3):
                traces[j] = traces[j].slice(starttime=starttime)

        if not (traces[0].stats.endtime == traces[1].stats.endtime
                and traces[0].stats.endtime == traces[2].stats.endtime):
            endtime = min([traces[0].stats.endtime,
                           traces[1].stats.endtime,
                           traces[2].stats.endtime])
            for j in range(3):
                traces[j] = traces[j].slice(endtime=endtime)

        start_flag = -1
        end_flag = -1
        event_list = []

        for windowed_st in traces.slide(window_length=(config.winsize - 1) / 100.0,
                                        step=config.winlag / 100.0):
            data_input = []
            for j in range(3):
                data_input.append(windowed_st[j].data)

            if model_name == 'cnn':
                # raw_data = [data_preprocess(d, 'bandpass', False) for d in data_input]
                data_input = np.array(data_input).T
                if preprocess:
                    # data_input = sklearn.preprocessing.minmax_scale(data_input)

                    data_mean = np.mean(data_input, axis=0)
                    data_input = np.absolute(data_input - data_mean)
                    data_input = data_input / (np.max(data_input, axis=0) + np.array([1, 1, 1]))
                data_input = np.array([np.array([data_input])])
            elif model_name == 'cldnn':
                # raw_data = [data_preprocess(d, 'bandpass', False) for d in data_input]

                data_input = [data_preprocess(d) for d in data_input]
                data_input = np.array(data_input).T
                data_input = np.array([data_input])

            class_pred, confidence = model.classify(sess=sess, input_=data_input)
            if class_pred == 1:

                # plt.subplot(3, 1, 1)
                # plt.plot(raw_data[0])
                # plt.subplot(3, 1, 2)
                # plt.plot(raw_data[1])
                # plt.subplot(3, 1, 3)
                # plt.plot(raw_data[2])
                # plt.show()


                if start_flag == -1:
                    start_flag = windowed_st[0].stats.starttime
                    end_flag = windowed_st[0].stats.endtime
                else:
                    end_flag = windowed_st[0].stats.endtime

            if class_pred == 0 and start_flag != -1 and end_flag < windowed_st[0].stats.starttime:
                event = [file[0].split('\\')[-1][:-4],
                         start_flag,
                         end_flag,
                         confidence]

                # print(event)


                event_list.append(event)
                start_flag = -1
                end_flag = -1

        if len(event_list) != 0:
            with open('detect_result/' + model_name + '/events_test.csv', mode='a', newline='') as f:
                csvwriter = csv.writer(f)
                for event in event_list:
                    csvwriter.writerow(event)
                f.close()

        start_point += 1
        with open('detect_result/' + model_name + '/checkpoint', mode='w') as f:
            f.write(str(start_point))
            end = datetime.datetime.now()
            print('{} scanned, num {}, time {}.'.format(file[0].split('\\')[-1][:-4], start_point, end - begin))
            print('checkpoint saved.')


if __name__ == '__main__':
    main('cnn', new_scan=True)
    # main('cldnn', new_scan=False)
