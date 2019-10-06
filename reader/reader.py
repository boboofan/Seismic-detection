# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 22:24:56 2017

@author: USTClinjf
"""

import os.path
import random
import numpy as np
import obspy
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import scipy.io as sio
import multiprocessing
from obspy.signal.filter import bandpass

from config.config import Config


def data_preprocess(data, filter_model='bandpass'):
    if filter_model == 'bandpass':
        data = bandpass(data, freqmin=4, freqmax=15, df=100)

    data = (data - np.mean(data)) / (np.max(np.absolute(data)) + 1)
    data = np.absolute(data)
    return data


def mlp_read_sac_data_for_cldnn(file_name_list,
                                sample=None,
                                preprocess=data_preprocess):
    neg_win_size = 60 * 100
    st = [obspy.read(file_name)[0].detrend('spline', order=2, dspline=50) for file_name in file_name_list]

    if not (st[0].stats.starttime == st[1].stats.starttime
            and st[0].stats.starttime == st[2].stats.starttime):
        starttime = max([st[0].stats.starttime,
                         st[1].stats.starttime,
                         st[2].stats.starttime])
        for j in range(3):
            st[j] = st[j].slice(starttime)

    if not (st[0].stats.endtime == st[1].stats.endtime
            and st[0].stats.endtime == st[2].stats.endtime):
        endtime = min([st[0].stats.endtime,
                       st[1].stats.endtime,
                       st[2].stats.endtime])
        for j in range(3):
            st[j] = st[j].slice(endtime=endtime)

    st_len = st[0].stats.npts

    if sample is None:
        x = [preprocess(i.data) for i in st]
        y = 1
        x = np.array(x).T
        data = [{'x': x, 'y': y}]
    else:
        st_x = np.array([i.data for i in st])
        data = []
        for point in sample:
            point = int(round(point * (st_len - neg_win_size)))
            tmp_x = st_x[:, point:point + neg_win_size]
            tmp_x = np.array([preprocess(i) for i in tmp_x]).T
            tmp_y = 0
            data.append({'x': tmp_x, 'y': tmp_y})

    return data


def mlp_read_sac_data_for_birnn(file_name_list,
                                sample=None,
                                preprocess=data_preprocess):
    neg_win_size = 60 * 100
    # for file_name in file_name_list:
    # print(file_name)
    st = [obspy.read(file_name)[0].detrend('spline', order=2, dspline=50) for file_name in file_name_list]
    # st = [obspy.read(file_name)[0] for file_name in file_name_list]

    if not (st[0].stats.starttime == st[1].stats.starttime
            and st[0].stats.starttime == st[2].stats.starttime):
        starttime = max([st[0].stats.starttime,
                         st[1].stats.starttime,
                         st[2].stats.starttime])
        for j in range(3):
            st[j] = st[j].slice(starttime)

    if not (st[0].stats.endtime == st[1].stats.endtime
            and st[0].stats.endtime == st[2].stats.endtime):
        endtime = min([st[0].stats.endtime,
                       st[1].stats.endtime,
                       st[2].stats.endtime])
        for j in range(3):
            st[j] = st[j].slice(endtime=endtime)

    st_len = st[0].stats.npts

    if sample is None:
        p_start_time = -1
        if hasattr(st[0].stats.sac, 't0'):
            p_start_time = st[0].stats.sac.t0 - st[0].stats.sac.b
            p_start_time = int(p_start_time * 100)

        s_start_time = -1
        if hasattr(st[0].stats.sac, 't1'):
            if st[0].stats.sac.t1 != -12345 and st[0].stats.sac.t1 != st[0].stats.sac.t0:
                s_start_time = st[0].stats.sac.t1 - st[0].stats.sac.b
                s_start_time = int(s_start_time * 100)

        # x = [preprocess(i.data) for i in st]
        x = [preprocess(i.data, filter_model='None') for i in st]
        y = np.zeros([st_len])
        x = np.array(x).T
        if p_start_time != -1:
            y[p_start_time:] = 1

        if s_start_time != -1:
            y[s_start_time:] = 2
        data = [{'x': x, 'y': y}]
    else:
        st_x = np.array([i.data for i in st])
        data = []
        for point in sample:
            point = int(round(point * (st_len - neg_win_size)))
            tmp_x = st_x[:, point:point + neg_win_size]
            tmp_x = np.array([preprocess(i) for i in tmp_x]).T
            # tmp_x = np.array([preprocess(i, filter_model='None') for i in tmp_x]).T
            tmp_y = np.zeros([neg_win_size])
            data.append({'x': tmp_x, 'y': tmp_y})

    return data


class Reader(object):
    def __init__(self):
        self.config = Config()
        self.use_cpu_rate = self.config.use_cpu_rate
        self.foledername = self.config.data_foldername
        self.winsize = self.config.winsize
        self.beforename = self.get_filename('before')
        self.aftername = self.get_filename('after')
        self.examplename = self.get_filename('example')
        self.events_examplename = self.get_filename('events_example')
        self.microSeismic = self.get_filename('microSeismic')
        # self.new_events = self.get_filename('new_events')

    def get_filename(self, dataset_type):
        if dataset_type == 'after':
            filename = os.path.join(self.config.re_data_foldername, dataset_type)
        else:
            filename = os.path.join(self.foledername, dataset_type)
        filename_dict = dict()
        if os.path.exists(filename):
            name_list = os.listdir(filename)
            for name in name_list:
                if dataset_type == 'microSeismic':
                    first_file_name = os.path.join(filename, name)
                    station_list = os.listdir(first_file_name)
                    for station in station_list:
                        if dataset_type == 'microSeismic':
                            second_file_name = os.path.join(first_file_name, station)
                            event_list = os.listdir(second_file_name)
                            for event in event_list:
                                station_name = event.split('.')[0]
                                name_key = name + station + station_name
                                if filename_dict.get(name_key) == None:
                                    filename_dict[name_key] = [os.path.join(second_file_name, event)]
                                else:
                                    filename_dict[name_key].append(os.path.join(second_file_name, event))

                        else:
                            station_name = station.split('.')[0]
                            name_key = name + station_name
                            if filename_dict.get(name_key) == None:
                                filename_dict[name_key] = [os.path.join(first_file_name, station)]
                            else:
                                filename_dict[name_key].append(os.path.join(first_file_name, station))

                else:
                    [name_key, comp] = os.path.splitext(name)
                    if dataset_type == 'example' or dataset_type == 'events_example':
                        [name_key, comp] = os.path.splitext(name_key)
                    elif dataset_type == 'new_events':
                        [name_key, comp] = os.path.splitext(name_key)
                    else:
                        name_key = name_key[:18]

                    if filename_dict.get(name_key) == None:
                        if dataset_type == 'after':
                            filename_dict[name_key] = [os.path.join(self.config.re_data_foldername, dataset_type, name)]
                        else:
                            filename_dict[name_key] = [os.path.join(self.foledername, dataset_type, name)]
                    else:
                        if dataset_type == 'after':
                            filename_dict[name_key].append(
                                os.path.join(self.config.re_data_foldername, dataset_type, name))
                        else:
                            filename_dict[name_key].append(os.path.join(self.foledername, dataset_type, name))

        else:
            print('{} is not exist.'.format(filename))
            return None

        filename_list = list(filename_dict.values())
        return filename_list

    def get_next_batch(self):
        return next(self.get_filebatch())

    def get_filebatch(self):
        file_batch_num = self.config.file_batch_num
        after_per_num = self.config.after_per_num
        before_per_num = self.config.before_per_num

        aftername = self.aftername
        beforename = self.beforename
        examplename = self.examplename

        for i in aftername:
            i.append(np.random.rand(after_per_num))

        for i in beforename:
            i.append(np.random.rand(before_per_num))

        for i in examplename:
            i.append(None)

        filename = aftername + beforename + examplename

        file_len = len(filename)
        file_i = 0
        while True:
            if file_i == 0:
                random.shuffle(filename)

            if file_i + file_batch_num > file_len:
                file_i = file_len - file_batch_num

            file_list = list()
            sample_point = list()

            for i in range(file_batch_num):
                file_list.append(filename[file_i + i][:-1])
                sample_point.append(filename[file_i + i][-1])

            batch_data = self.read_sac_data(file_list, sample_point=sample_point)
            yield batch_data

            file_i += file_batch_num
            if file_i == file_len:
                file_i = 0

    def read_sac_data(self, file_list, sample_point=None, normalize=True, downsample=None, isevent=False):
        file_len = len(file_list)
        if sample_point is not None:
            if file_len != len(sample_point):
                print('The number of file_list is not match the sample_point!')
                return None

        batch_data = list()
        for i in range(file_len):
            traces = list()
            for j in range(3):
                tmp_filename = os.path.join(self.foledername, file_list[i][j])
                tmp_trace = obspy.read(tmp_filename)[0]
                if downsample is not None:
                    tmp_trace.decimate(downsample)
                traces.append(tmp_trace)

            if not (traces[0].stats.starttime == traces[1].stats.starttime
                    and traces[0].stats.starttime == traces[2].stats.starttime):
                starttime = max([traces[0].stats.starttime,
                                 traces[1].stats.starttime,
                                 traces[2].stats.starttime])
                for j in range(3):
                    traces[j] = traces[j].slice(starttime)

            if not (traces[0].stats.endtime == traces[1].stats.endtime
                    and traces[0].stats.endtime == traces[2].stats.endtime):
                endtime = min([traces[0].stats.endtime,
                               traces[1].stats.endtime,
                               traces[2].stats.endtime])
                for j in range(3):
                    traces[j] = traces[j].slice(endtime=endtime)

            trace_len = traces[0].stats.npts
            p_start = -1
            s_start = -1
            if hasattr(traces[0].stats.sac, 't0'):
                p_start = int(traces[0].stats.sac.t0 * 100)
            if hasattr(traces[0].stats.sac, 't1'):
                s_start = int(traces[0].stats.sac.t1 * 100)

            for j in range(3):
                traces[j] = traces[j].data
            traces = np.array(traces)

            if trace_len < self.winsize:
                traces = np.concatenate((traces, np.zeros([3, self.winsize - trace_len])), axis=1)

            if sample_point is None or sample_point[i] is None:
                if isevent:
                    if s_start != -1:
                        if s_start < self.winsize:
                            tmp_data = traces[:, :self.winsize].T
                        else:
                            tmp_end = min([s_start + 300, trace_len - 1])
                            tmp_data = traces[:, tmp_end - self.winsize + 1: tmp_end + 1].T
                    else:
                        if p_start < 500:
                            tmp_data = traces[:, :self.winsize].T
                        else:
                            tmp_start = min([p_start - 400, trace_len - self.winsize - 1])
                            tmp_data = traces[:, tmp_start: tmp_start + self.winsize].T

                else:
                    tmp_data = traces[:, :self.winsize].T

                if normalize:
                    # tmp_data = sklearn.preprocessing.minmax_scale(tmp_data)
                    tmp_mean = np.mean(tmp_data, axis=0)
                    tmp_data = np.absolute(tmp_data - tmp_mean)
                    tmp_data = tmp_data / (np.max(tmp_data, axis=0) + np.array([1, 1, 1]))
                batch_data.append(np.array([tmp_data]))
                continue

            for point in sample_point[i]:
                point = int(round(point * (trace_len - self.winsize)))

                tmp_data = traces[:, point:point + self.winsize].T
                if normalize:
                    # tmp_data = sklearn.preprocessing.minmax_scale(tmp_data)
                    tmp_mean = np.mean(tmp_data, axis=0)
                    tmp_data = np.absolute(tmp_data - tmp_mean)
                    tmp_data = tmp_data / (np.max(tmp_data, axis=0) + np.array([1, 1, 1]))
                batch_data.append(np.array([tmp_data]))

        batch_data = np.array(batch_data)
        return batch_data

    def get_MLT_train_data(self, normalize=True, encode=True, save_mat_file=None):
        from autoencoder.autoencoder import AutoEncoder

        example_file_list = self.examplename
        noise_file_list = self.beforename

        noise_file_dict = dict()
        for i in noise_file_list:
            station_name = i[0].split('\\')[1].split('.')[1]
            if station_name not in noise_file_dict:
                noise_file_dict[station_name] = [i]
            else:
                noise_file_dict[station_name].append(i)

        noise_file_list = []
        for i in noise_file_dict.keys():
            tmp = random.sample(noise_file_dict[i], 3)
            for j in tmp:
                noise_file_list.append(j)

        training_data = dict()
        training_label = dict()

        example_data = self.read_sac_data(example_file_list, normalize=normalize)
        if len(example_file_list) != len(example_data):
            print('read_sac_data error!')
            return None
        for i in range(len(example_file_list)):
            station_name = example_file_list[i][0].split('\\')[1].split('.')[1]
            if station_name not in training_data:
                training_data[station_name] = [example_data[i]]
                training_label[station_name] = [1]
            else:
                training_data[station_name].append(example_data[i])
                training_label[station_name].append(1)

        noise_sample_point = list()
        for i in range(len(noise_file_list)):
            noise_sample_point.append([np.random.rand()])

        noise_data = self.read_sac_data(noise_file_list, sample_point=noise_sample_point, normalize=normalize)
        for i in range(len(noise_file_list)):
            station_name = noise_file_list[i][0].split('\\')[1].split('.')[1]
            if station_name not in training_data:
                training_data[station_name] = [noise_data[i]]
                training_label[station_name] = [0]
            else:
                training_data[station_name].append(noise_data[i])
                training_label[station_name].append(0)

        if encode == True:
            autoenc = AutoEncoder()

            for key in training_data.keys():
                tmp = np.array(training_data[key])
                tmp = autoenc.encode(tmp)
                training_data[key] = tmp

        if save_mat_file is not None:
            sio.savemat(save_mat_file, {'training_data': training_data,
                                        'training_label': training_label})

        return training_data, training_label

    def get_cnn_batch_data(self, data_type):
        train_pos_file_name, test_pos_file_name = train_test_split(self.microSeismic,
                                                                   test_size=0.2)
        train_neg_file_name, test_neg_file_name = train_test_split(self.beforename,
                                                                   test_size=0.2)
        train_pos_num = len(train_pos_file_name)
        train_neg_num = len(train_neg_file_name)
        pos_batch_num = self.config.cnn_pos_batch_num
        neg_file_batch_num = self.config.cnn_neg_file_batch_num
        neg_file_per_num = self.config.cnn_neg_file_per_num

        def get_batch():
            pos_batch_i = 0
            neg_batch_i = 0
            while True:
                if pos_batch_i == 0:
                    random.shuffle(train_pos_file_name)

                if neg_batch_i == 0:
                    random.shuffle(train_neg_file_name)

                if pos_batch_i + pos_batch_num > train_pos_num:
                    pos_batch_i = train_pos_num - pos_batch_num

                if neg_batch_i + neg_file_batch_num > train_neg_num:
                    neg_batch_i = train_neg_num - neg_file_batch_num

                pos_x = self.read_sac_data(train_pos_file_name[pos_batch_i: pos_batch_i + pos_batch_num],
                                           isevent=True)
                batch_y = np.ones(len(pos_x), dtype=int)
                batch_x = pos_x

                neg_x = self.read_sac_data(train_neg_file_name[neg_batch_i: neg_batch_i + neg_file_batch_num],
                                           sample_point=np.random.rand(neg_file_batch_num, neg_file_per_num))
                batch_y = np.concatenate((batch_y, np.zeros(len(neg_x), dtype=int)))
                batch_x = np.concatenate((batch_x, neg_x))
                yield batch_x, batch_y

                pos_batch_i += pos_batch_num
                neg_batch_i += neg_file_batch_num

                if pos_batch_i == train_pos_num:
                    pos_batch_i = 0
                if neg_batch_i == train_neg_num:
                    neg_batch_i = 0

        if data_type == 'train':
            return next(get_batch())

        if data_type == 'test':
            sample_pos_file_name = random.sample(test_pos_file_name, 50)
            sample_neg_file_name = random.sample(test_neg_file_name, 20)

            test_pos_x = self.read_sac_data(sample_pos_file_name, downsample=2)
            test_pos_y = np.ones(len(test_pos_x), dtype=int)

            test_neg_x = self.read_sac_data(sample_neg_file_name,
                                            sample_point=np.random.rand(len(sample_neg_file_name), neg_file_per_num))
            test_y = np.concatenate((test_pos_y, np.zeros(len(test_neg_x), dtype=int)))
            test_x = np.concatenate((test_pos_x, test_neg_x))
            return test_x, test_y

    def get_birnn_batch_data(self, data_type):
        def get_batch():
            pos_batch_i = 0
            neg_batch_i = 0
            while True:
                if pos_batch_i == 0:
                    random.shuffle(train_pos_file_name)

                if neg_batch_i == 0:
                    random.shuffle(train_neg_file_name)

                if pos_batch_i + pos_batch_num > train_pos_num:
                    pos_batch_i = train_pos_num - pos_batch_num

                if neg_batch_i + neg_file_batch_num > train_neg_num:
                    neg_batch_i = train_neg_num - neg_file_batch_num

                batch_pos_file_name = train_pos_file_name[pos_batch_i: pos_batch_i + pos_batch_num]
                batch_neg_file_name = train_neg_file_name[neg_batch_i: neg_batch_i + neg_file_batch_num]

                cores = int(multiprocessing.cpu_count() * self.use_cpu_rate)
                pool = multiprocessing.Pool(processes=cores)

                pool_list = []
                for pos_file_name_list in batch_pos_file_name:
                    pool_list.append(pool.apply_async(mlp_read_sac_data_for_birnn, args=(pos_file_name_list,
                                                                                         None,
                                                                                         data_preprocess)))

                for neg_file_name_list in batch_neg_file_name:
                    tmp_rand = np.random.rand(neg_file_per_num)
                    pool_list.append(pool.apply_async(mlp_read_sac_data_for_birnn, args=(neg_file_name_list,
                                                                                         tmp_rand,
                                                                                         data_preprocess)))

                pool.close()
                pool.join()

                batch_data = []

                for i in pool_list:
                    tmp_data = i.get()
                    for j in tmp_data:
                        batch_data.append(j)

                random.shuffle(batch_data)

                batch_x = []
                batch_y = []
                for data in batch_data:
                    batch_x.append(data['x'])
                    batch_y.append(data['y'])

                batch_x = np.array(batch_x)
                batch_y = np.array(batch_y)

                yield batch_x, batch_y

                pos_batch_i += pos_batch_num
                neg_batch_i += neg_file_batch_num

                if pos_batch_i == train_pos_num:
                    pos_batch_i = 0
                if neg_batch_i == train_neg_num:
                    neg_batch_i = 0

        # train_pos_file_name, test_pos_file_name = train_test_split(self.microSeismic + self.examplename,
        #                                                            test_size=0.25,
        #                                                            random_state=5)
        # train_neg_file_name, test_neg_file_name = train_test_split(self.beforename,
        #                                                            test_size=0.25,
        #                                                            random_state=5)

        train_pos_file_name, test_pos_file_name = train_test_split(self.examplename,
                                                                   test_size=0.1,
                                                                   random_state=5)
        train_neg_file_name, test_neg_file_name = train_test_split(self.beforename,
                                                                   test_size=0.1,
                                                                   random_state=5)

        train_pos_num = len(train_pos_file_name)
        train_neg_num = len(train_neg_file_name)
        pos_batch_num = self.config.birnn_pos_batch_num
        neg_file_batch_num = self.config.birnn_neg_file_batch_num
        neg_file_per_num = self.config.birnn_neg_file_per_num

        if data_type == 'train':
            return next(get_batch())

        if data_type == 'test':
            sample_pos_file_name = random.sample(test_pos_file_name, pos_batch_num * 2)
            sample_neg_file_name = random.sample(test_neg_file_name, neg_file_batch_num * 2)

            cores = int(multiprocessing.cpu_count() * self.use_cpu_rate)
            pool = multiprocessing.Pool(processes=cores)

            pool_list = []
            for file_name_list in sample_pos_file_name:
                pool_list.append(pool.apply_async(mlp_read_sac_data_for_birnn, args=(file_name_list,
                                                                                     None,
                                                                                     data_preprocess)))

            for file_name_list in sample_neg_file_name:
                tmp_rand = np.random.rand(neg_file_per_num)
                pool_list.append(pool.apply_async(mlp_read_sac_data_for_birnn, args=(file_name_list,
                                                                                     tmp_rand,
                                                                                     data_preprocess)))

            pool.close()
            pool.join()

            test_data = []

            for i in pool_list:
                tmp_data = i.get()
                for j in tmp_data:
                    test_data.append(j)

            random.shuffle(test_data)

            test_x = []
            test_y = []
            for data in test_data:
                test_x.append(data['x'])
                test_y.append(data['y'])

            test_x = np.array(test_x)
            test_y = np.array(test_y)
            return test_x, test_y

    def get_cldnn_batch_data(self, data_type):
        def get_batch():
            pos_batch_i = 0
            neg_batch_i = 0
            while True:
                if pos_batch_i == 0:
                    random.shuffle(train_pos_file_name)

                if neg_batch_i == 0:
                    random.shuffle(train_neg_file_name)

                if pos_batch_i + pos_batch_num > train_pos_num:
                    pos_batch_i = train_pos_num - pos_batch_num

                if neg_batch_i + neg_file_batch_num > train_neg_num:
                    neg_batch_i = train_neg_num - neg_file_batch_num

                batch_pos_file_name = train_pos_file_name[pos_batch_i: pos_batch_i + pos_batch_num]
                batch_neg_file_name = train_neg_file_name[neg_batch_i: neg_batch_i + neg_file_batch_num]

                cores = int(multiprocessing.cpu_count() * self.use_cpu_rate)
                pool = multiprocessing.Pool(processes=cores)

                pool_list = []
                for pos_file_name_list in batch_pos_file_name:
                    pool_list.append(pool.apply_async(mlp_read_sac_data_for_cldnn, args=(pos_file_name_list,
                                                                                         None,
                                                                                         data_preprocess)))

                for neg_file_name_list in batch_neg_file_name:
                    tmp_rand = np.random.rand(neg_file_per_num)
                    pool_list.append(pool.apply_async(mlp_read_sac_data_for_cldnn, args=(neg_file_name_list,
                                                                                         tmp_rand,
                                                                                         data_preprocess)))

                pool.close()
                pool.join()

                batch_data = []

                for i in pool_list:
                    tmp_data = i.get()
                    for j in tmp_data:
                        batch_data.append(j)

                random.shuffle(batch_data)

                batch_x = []
                batch_y = []
                for data in batch_data:
                    batch_x.append(data['x'])
                    batch_y.append(data['y'])

                batch_x = np.array(batch_x)
                batch_y = np.array(batch_y)

                yield batch_x, batch_y

                pos_batch_i += pos_batch_num
                neg_batch_i += neg_file_batch_num

                if pos_batch_i == train_pos_num:
                    pos_batch_i = 0
                if neg_batch_i == train_neg_num:
                    neg_batch_i = 0

        train_pos_file_name, test_pos_file_name = train_test_split(self.microSeismic + self.events_examplename,
                                                                   test_size=0.25,
                                                                   random_state=5)
        train_neg_file_name, test_neg_file_name = train_test_split(self.beforename,
                                                                   test_size=0.25,
                                                                   random_state=5)
        train_pos_num = len(train_pos_file_name)
        train_neg_num = len(train_neg_file_name)
        pos_batch_num = self.config.cldnn_pos_batch_num
        neg_file_batch_num = self.config.cldnn_neg_file_batch_num
        neg_file_per_num = self.config.cldnn_neg_file_per_num

        if data_type == 'train':
            return next(get_batch())

        if data_type == 'test':
            sample_pos_file_name = random.sample(test_pos_file_name, pos_batch_num * 2)
            sample_neg_file_name = random.sample(test_neg_file_name, neg_file_batch_num * 2)

            cores = int(multiprocessing.cpu_count() * self.use_cpu_rate)
            pool = multiprocessing.Pool(processes=cores)

            pool_list = []
            for file_name_list in sample_pos_file_name:
                pool_list.append(pool.apply_async(mlp_read_sac_data_for_cldnn, args=(file_name_list,
                                                                                     None,
                                                                                     data_preprocess)))

            for file_name_list in sample_neg_file_name:
                tmp_rand = np.random.rand(neg_file_per_num)
                pool_list.append(pool.apply_async(mlp_read_sac_data_for_cldnn, args=(file_name_list,
                                                                                     tmp_rand,
                                                                                     data_preprocess)))

            pool.close()
            pool.join()

            test_data = []

            for i in pool_list:
                tmp_data = i.get()
                for j in tmp_data:
                    test_data.append(j)

            random.shuffle(test_data)

            test_x = []
            test_y = []
            for data in test_data:
                test_x.append(data['x'])
                test_y.append(data['y'])

            test_x = np.array(test_x)
            test_y = np.array(test_y)
            return test_x, test_y


if __name__ == '__main__':
    pass
