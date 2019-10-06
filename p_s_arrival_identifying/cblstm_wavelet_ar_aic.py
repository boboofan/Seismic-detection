import tensorflow as tf
import numpy as np
from obspy.signal.filter import bandpass
from spectrum import aryule
from phasepapy.phasepicker import aicdpicker
import csv
import obspy
import os
import multiprocessing
import matplotlib.pyplot as plt
import datetime
import math
import pywt
import copy

from p_s_arrival_identifying.cblstm import CBLSTM
from reader.reader import Reader
from config.config import Config
from tflib.models import Model
from p_s_arrival_identifying.fnd_sort import fnd_sort


def data_preprocess(data, filter_model='bandpass', is_abs=True):
    if filter_model == 'bandpass':
        data = bandpass(data, freqmin=3, freqmax=15, df=100)

    data = (data - np.mean(data)) / (np.max(np.absolute(data)) + 1)
    if is_abs:
        data = np.absolute(data)
    return data


def get_ar_aic(window, order=3, n_win=100, s_win=100):
    data_len = len(window[0])
    aic = np.zeros(data_len)
    for data in window:
        An = aryule(data[:n_win], order)[0]
        As = aryule(data[data_len - s_win:], order)[0]
        An = An * -1
        As = As * -1
        for i in range(data_len):
            if i <= order or i >= data_len - order:
                aic[i] = 1e7
                continue
            var_before = 0
            for j in range(order, i + 1):
                var_before += (data[j] - np.dot(An, data[j - order: j])) ** 2 / (i + 1 - order)
            var_after = 0
            for j in range(i + 1, data_len - order + 1):
                var_after += (data[j] - np.dot(As, data[j - order: j])) ** 2 / (data_len - order - i)
            aic[i] += (i - order) * np.log(var_before + 0.1) + (data_len - order - i) * np.log(var_after + 0.1)

        return aic


def get_energy(window):
    return np.sum(window ** 2)


def transform_submission_time(arrival_index, starttime):
    arrival_time = arrival_index / 100.0
    submission_time = float((starttime + 8 * 3600 + arrival_time).strftime('%Y%m%d%H%M%S.%f'))
    return '%.2f' % submission_time


def mlp_aic(st,
            stats,
            p_index,
            s_index,
            class_prob,
            ps_lwinsize,
            ps_rwinsize,
            min_p_s_time):
    try:
        data_len = stats[-1]
        st_raw = copy.deepcopy(st)

        bhz = st[2].filter('bandpass', freqmin=3, freqmax=15)
        bhn = st[1].filter('bandpass', freqmin=3, freqmax=15)
        bhe = st[0].filter('bandpass', freqmin=3, freqmax=15)

        bhz_max = np.max(np.abs(bhz.data))
        bhn_max = np.max(np.abs(bhn.data))
        bhe_max = np.max(np.abs(bhe.data))
        max_amplitude = np.max([bhz_max, bhn_max, bhe_max])

        tmp_bhe = np.zeros([data_len], dtype=np.float64)
        tmp_bhn = np.zeros([data_len], dtype=np.float64)
        tmp_bhz = np.zeros([data_len], dtype=np.float64)

        tmp_bhe[0] = bhe.data[0]
        tmp_bhn[0] = bhn.data[0]
        tmp_bhz[0] = bhz.data[0]

        for i in range(data_len):
            if i == 0:
                continue
            tmp_bhz[i] = bhz.data[i] ** 2 + 3 * (bhz.data[i] - bhz.data[i - 1]) ** 2
            tmp_bhn[i] = bhn.data[i] ** 2 + 3 * (bhn.data[i] - bhn.data[i - 1]) ** 2
            tmp_bhe[i] = bhe.data[i] ** 2 + 3 * (bhe.data[i] - bhe.data[i - 1]) ** 2

        bhz.data = tmp_bhz
        bhn.data = tmp_bhn
        bhe.data = tmp_bhe

        p_range_start = 75
        p_range_end = data_len - 75 - 1
        p_cfd = -1
        if p_index != -1:
            p_range_start = max([p_index * 100 - ps_lwinsize, p_range_start])
            p_range_end = min([(p_index + 1) * 100 + ps_rwinsize, p_range_end])
            if p_index + 1 < len(class_prob):
                p_cfd = (class_prob[p_index, 1] + class_prob[p_index + 1, 1]) / 2.0
            else:
                p_cfd = class_prob[p_index, 1]

            p_win = []
            if sum(st_raw[2].data[p_range_start: p_range_end] == 0) <= 150:
                p_win.append(st_raw[2].data[p_range_start: p_range_end])
            elif sum(st_raw[1].data[p_range_start: p_range_end] == 0) <= 150:
                p_win.append(st_raw[1].data[p_range_start: p_range_end])
            elif sum(st_raw[0].data[p_range_start: p_range_end + 1] == 0) <= 150:
                p_win.append(st_raw[0].data[p_range_start: p_range_end])
            if len(p_win) == 0:
                event_p_start = 0
                p_cfd = 0
            else:
                p_aic = get_ar_aic(p_win)
                event_p_start = p_range_start + np.argmin(p_aic)
        else:
            p_picker = aicdpicker.AICDPicker(t_ma=3, nsigma=5, t_up=0.78, nr_len=2, nr_coeff=2, pol_len=10,
                                             pol_coeff=10,
                                             uncert_coeff=3)
            p_bhz = st_raw[2].copy()
            if sum(p_bhz.data == 0) >= len(p_bhz.data) * 0.5:
                p_bhz = st_raw[0].copy()
            if sum(p_bhz.data == 0) >= len(p_bhz.data) * 0.5:
                p_bhz = st_raw[1].copy()

            _, p_picks, _, _, _ = p_picker.picks(p_bhz)
            if len(p_picks) != 0:
                event_p_start = p_picks[0]
                event_p_start = int((event_p_start - bhz.stats.starttime) * 100)
            else:
                p_energy_ratio = np.zeros([data_len], dtype=np.float32) - 1.0
                i = p_range_start
                while i <= p_range_end:
                    p_energy_ratio[i] = get_energy(bhz.data[i: i + 75 + 1]) / (get_energy(
                        bhz.data[i - 75: i + 1]) + 1e-12)
                    i += 1
                event_p_start = np.argmax(p_energy_ratio)

        if p_cfd == -1:
            first = math.floor(event_p_start / 100)
            if first + 1 < len(class_prob):
                p_cfd = (class_prob[first, 1] + class_prob[first + 1, 1]) / 2.0
            else:
                p_cfd = class_prob[first, 1]

        s_range_start = event_p_start + min_p_s_time
        s_range_end = data_len - 75 - 1
        s_cfd = -1
        if s_index != -1:
            s_range_start = max([s_index * 100 - ps_lwinsize, s_range_start])
            s_range_end = min([(s_index + 1) * 100 + ps_rwinsize, s_range_end])
            if s_range_end <= s_range_start + 250:
                s_range_start = s_range_end - 251
                if s_range_start < 0:
                    s_range_start = 0
                    s_range_end = 251
            if s_index + 1 < len(class_prob):
                s_cfd = (class_prob[s_index, 2] + class_prob[s_index + 1, 2]) / 2.0
            else:
                s_cfd = class_prob[s_index, 2]

            s_bhe = st_raw[0].data[s_range_start: s_range_end]
            s_bhn = st_raw[1].data[s_range_start: s_range_end]
            s_bhz = st_raw[2].data[s_range_start: s_range_end]

            s_bhe = pywt.wavedec(s_bhe, 'db2', level=3)
            s_bhe = [s_bhe[0], None, None, None]
            s_bhe = pywt.waverec(s_bhe, 'db2')
            s_bhn = pywt.wavedec(s_bhn, 'db2', level=3)
            s_bhn = [s_bhn[0], None, None, None]
            s_bhn = pywt.waverec(s_bhn, 'db2')
            s_bhz = pywt.wavedec(s_bhz, 'db2', level=3)
            s_bhz = [s_bhz[0], None, None, None]
            s_bhz = pywt.waverec(s_bhz, 'db2')

            tmp_bhe = np.zeros([len(s_bhe)], dtype=np.float64)
            tmp_bhn = np.zeros([len(s_bhe)], dtype=np.float64)
            tmp_bhz = np.zeros([len(s_bhe)], dtype=np.float64)

            tmp_bhe[0] = s_bhe.data[0]
            tmp_bhn[0] = s_bhn.data[0]
            tmp_bhz[0] = s_bhz.data[0]

            for i in range(len(s_bhe)):
                if i == 0:
                    continue
                tmp_bhn[i] = s_bhn[i] ** 2 + 3 * (s_bhn[i] - s_bhn[i - 1]) ** 2
                tmp_bhe[i] = s_bhe[i] ** 2 + 3 * (s_bhe[i] - s_bhe[i - 1]) ** 2
                tmp_bhz[i] = s_bhz[i] ** 2 + 3 * (s_bhz[i] - s_bhz[i - 1]) ** 2

            s_bhn = tmp_bhn
            s_bhe = tmp_bhe
            s_bhz = tmp_bhz


            s_n_pick = 10000000
            s_e_pick = 10000000
            s_z_pick = 10000000
            if sum(s_bhn == 0) <= 150:
                s_n_aic = get_ar_aic([s_bhn], n_win=100, s_win=100)
                s_n_pick = np.argmin(s_n_aic)
            if sum(s_bhe == 0) <= 150:
                s_e_aic = get_ar_aic([s_bhe], n_win=100, s_win=100)
                s_e_pick = np.argmin(s_e_aic)
            if sum(s_bhz == 0) <= 150:
                s_z_aic = get_ar_aic([s_bhz], n_win=100, s_win=100)
                s_z_pick = np.argmin(s_z_aic)
            s_pick = np.min([s_e_pick, s_n_pick, s_z_pick])
            if s_pick == 10000000:
                event_s_start = 0
                s_cfd = 0
            else:
                event_s_start = s_range_start + s_pick
        else:
            s_picker = aicdpicker.AICDPicker(t_ma=3, nsigma=5, t_up=0.78, nr_len=2, nr_coeff=2, pol_len=10,
                                             pol_coeff=10,
                                             uncert_coeff=3)
            s_bhn = bhn.copy()
            s_bhe = bhe.copy()
            if sum(s_bhn.data == 0) >= len(s_bhn.data) * 0.5 and sum(s_bhe.data == 0) >= len(s_bhe.data) * 0.5:
                s_bhn = bhz.copy()
                s_bhe = bhz.copy()

            _, s_n_picks, _, _, s_n_uncert = s_picker.picks(s_bhn)
            _, s_e_picks, _, _, s_e_uncert = s_picker.picks(s_bhe)
            if len(s_n_picks) != 0 or len(s_e_picks) != 0:
                s_n_pick = None
                s_e_pick = None
                if len(s_n_picks) != 0:
                    s_n_pick = s_n_picks[np.argmin(s_n_uncert)]
                if len(s_e_picks) != 0:
                    s_e_pick = s_e_picks[np.argmin(s_e_uncert)]
                if s_n_pick is None:
                    event_s_start = s_e_pick
                elif s_e_pick is None:
                    event_s_start = s_n_pick
                else:
                    event_s_start = min([s_n_pick, s_e_pick])

                event_s_start = int((event_s_start - bhz.stats.starttime) * 100)

            else:
                s_energy_ratio = np.zeros([data_len], dtype=np.float32) - 1.0
                i = s_range_start
                while i <= s_range_end:
                    bhe_energy_ratio = get_energy(bhe.data[i: i + 75] + 1) / (get_energy(
                        bhe.data[i - 75: i + 1]) + 1e-12)
                    bhn_energy_ratio = get_energy(bhn.data[i: i + 75] + 1) / (get_energy(
                        bhn.data[i - 75: i + 1]) + 1e-12)
                    s_energy_ratio[i] = (bhe_energy_ratio + bhn_energy_ratio) / 2.0
                    i += 1
                event_s_start = np.argmax(s_energy_ratio)

        if s_cfd == -1:
            first = math.floor(event_s_start / 100)
            if first + 1 < len(class_prob):
                s_cfd = (class_prob[first, 2] + class_prob[first + 1, 2]) / 2.0
            else:
                s_cfd = class_prob[first, 2]

        if event_p_start + 100 + 1 >= len(bhz.data):
            event_p_start = len(bhz.data) - 100 - 2
        if event_p_start - 100 < 0:
            event_p_start = 100
        snr = get_energy(st_raw[2].data[event_p_start: event_p_start + 100 + 1]) / (get_energy(
            st_raw[2].data[event_p_start - 100: event_p_start + 1]) + 1e-12)

        if np.std(st_raw[2].data[event_p_start - 100:event_p_start]) <= 3:
            snr = 0

        if event_s_start + 100 + 1 >= len(bhe.data):
            event_s_start = len(bhe.data) - 100 - 2
        if event_s_start - 100 < 0:
            event_s_start = 100

        s_e_snr = get_energy(st_raw[0].data[event_s_start: event_s_start + 100 + 1]) / (get_energy(
            st_raw[0].data[event_s_start - 100: event_s_start + 1]) + 1e-12)
        s_n_snr = get_energy(st_raw[1].data[event_s_start: event_s_start + 100 + 1]) / (get_energy(
            st_raw[1].data[event_s_start - 100: event_s_start + 1]) + 1e-12)
        s_z_snr = get_energy(st_raw[2].data[event_s_start: event_s_start + 100 + 1]) / (get_energy(
            st_raw[2].data[event_s_start - 100: event_s_start + 1]) + 1e-12)
        s_snr = np.max([s_e_snr, s_n_snr, s_z_snr])
        if s_snr >= 3:
            s_snr = snr
        else:
            s_snr = 0


        if p_index != -1 and s_index != -1:
            if event_s_start - event_p_start >= 2000:
                s_snr = 0
                snr = 0


        # print('snr:{}, s_snr:{}, max_amp:{}.'.format(snr, s_snr, max_amplitude))
        # print('p_cfd:{}, s_cfd:{}'.format(p_cfd, s_cfd))
        # plt.subplot(4, 1, 1)
        # plt.plot(st_raw[0].data, 'k')
        # plt.axvline(event_p_start, color='b')
        # plt.axvline(event_s_start, color='g')
        # plt.axvline(p_range_start, color='r', ls='--')
        # plt.axvline(p_range_end, color='r', ls='--')
        # plt.axvline(s_range_start, color='r', ls='--')
        # plt.axvline(s_range_end, color='r', ls='--')
        # # plt.subplot(7, 1, 2)
        # # plt.plot(bhe.data, 'k')
        # # plt.axvline(event_p_start, color='b')
        # # plt.axvline(event_s_start, color='g')
        # # plt.axvline(p_range_start, color='r', ls='--')
        # # plt.axvline(p_range_end, color='r', ls='--')
        # # plt.axvline(s_range_start, color='r', ls='--')
        # # plt.axvline(s_range_end, color='r', ls='--')
        # plt.subplot(4, 1, 2)
        # plt.plot(st_raw[1].data, 'k')
        # plt.axvline(event_p_start, color='b')
        # plt.axvline(event_s_start, color='g')
        # plt.axvline(p_range_start, color='r', ls='--')
        # plt.axvline(p_range_end, color='r', ls='--')
        # plt.axvline(s_range_start, color='r', ls='--')
        # plt.axvline(s_range_end, color='r', ls='--')
        # # plt.subplot(7, 1, 4)
        # # plt.plot(bhn.data, 'k')
        # # plt.axvline(event_p_start, color='b')
        # # plt.axvline(event_s_start, color='g')
        # # plt.axvline(p_range_start, color='r', ls='--')
        # # plt.axvline(p_range_end, color='r', ls='--')
        # # plt.axvline(s_range_start, color='r', ls='--')
        # # plt.axvline(s_range_end, color='r', ls='--')
        # plt.subplot(4, 1, 3)
        # plt.plot(st_raw[2].data, 'k')
        # plt.axvline(event_p_start, color='b')
        # plt.axvline(event_s_start, color='g')
        # plt.axvline(p_range_start, color='r', ls='--')
        # plt.axvline(p_range_end, color='r', ls='--')
        # plt.axvline(s_range_start, color='r', ls='--')
        # plt.axvline(s_range_end, color='r', ls='--')
        # # plt.subplot(7, 1, 6)
        # # plt.plot(bhz.data, 'k')
        # # plt.axvline(event_p_start, color='b')
        # # plt.axvline(event_s_start, color='g')
        # # plt.axvline(p_range_start, color='r', ls='--')
        # # plt.axvline(p_range_end, color='r', ls='--')
        # # plt.axvline(s_range_start, color='r', ls='--')
        # # plt.axvline(s_range_end, color='r', ls='--')
        # plt.subplot(4, 1, 4)
        # plt.plot(class_prob[: int(data_len // 100) + 1, 0], 'b', label='noise')
        # plt.plot(class_prob[: int(data_len // 100) + 1, 1], 'r', label='P_prob')
        # plt.plot(class_prob[: int(data_len // 100) + 1, 2], 'k', label='S_prob')
        # plt.show()


        p_submission_time = transform_submission_time(event_p_start, stats[1])
        s_submission_time = transform_submission_time(event_s_start, stats[1])
        p_submission = [stats[0], p_submission_time, 'P', p_cfd, snr, max_amplitude]
        s_submission = [stats[0], s_submission_time, 'S', s_cfd, s_snr, max_amplitude]
    except Exception as e:
        print('error!!!')
        print(('error type: {}'.format(e)))
        return []

    return p_submission, s_submission


def get_p_s_start_time(events):
    tmp_path = os.path.join(data_path, 'after', events[0][0])
    event_station = events[0][0].split('.')[1]
    st_bhz = obspy.read(tmp_path + '.BHZ')[0]
    st_bhe = obspy.read(tmp_path + '.BHE')[0]
    st_bhn = obspy.read(tmp_path + '.BHN')[0]
    st = [st_bhe, st_bhn, st_bhz]

    events_data = []
    events_stats = []
    events_raw = []

    for event in events:
        event_start = obspy.UTCDateTime(event[1])
        event_end = obspy.UTCDateTime(event[2]) + 15

        event_end = min([event_end,
                         st[0].stats.endtime,
                         st[1].stats.endtime,
                         st[2].stats.endtime])
        event_st = [i.slice(event_start, event_end).detrend('spline', order=2, dspline=50) for i in st]
        events_raw.append(event_st)
        event_data_len = len(event_st[0].data)
        event_data = np.array([data_preprocess(i.data, is_abs=True) for i in event_st]).T
        events_data.append(event_data)
        events_stats.append([event_station, event_start, event_end, event_data_len])

    events_data = np.array(events_data)

    events_p_index, events_s_index, events_class_prob = model.pickup_p_s(sess, events_data)

    cores = int(multiprocessing.cpu_count() * use_cpu_rate)
    pool = multiprocessing.Pool(processes=cores)

    event_submission_result = list()

    for i, event_st in enumerate(events_raw):
        event_submission_result.append(pool.apply_async(mlp_aic, args=(event_st,
                                                                       events_stats[i],
                                                                       events_p_index[i],
                                                                       events_s_index[i],
                                                                       events_class_prob[i],
                                                                       ps_lwinsize,
                                                                       ps_rwinsize,
                                                                       min_p_s_time)))

    pool.close()
    pool.join()

    event_submission = list()
    for i in event_submission_result:
        tmp_list = i.get()
        if len(tmp_list) == 0:
            continue
        p_submission, s_submission = tmp_list
        event_submission.append(p_submission)
        event_submission.append(s_submission)

    return event_submission


def cblstm_aic(events_file,
               output_file='submission.csv'):
    events_list = []
    with open(events_file) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) == 0:
                continue

            events_list.append(row)

    events_dict = dict()
    for event in events_list:
        if events_dict.get(event[0]) == None:
            events_dict[event[0]] = [event]
        else:
            events_dict[event[0]].append(event)

    event_submission_result = list()
    i = 0
    total = len(events_dict.keys())
    print('total file num: {}'.format(total))
    keys_list = list(events_dict.keys())
    # np.random.shuffle(keys_list)
    for key in keys_list:
        begin = datetime.datetime.now()
        i += 1
        event_submission_result.append(get_p_s_start_time(events_dict[key]))
        end = datetime.datetime.now()
        print('file {}, num {} done, time: {}'.format(events_dict[key][0][0], i, end - begin))

    event_submission = list()
    for i in range(total):
        tmp_list = event_submission_result[i]
        for j in tmp_list:
            if is_fnd_sort:
                if j[3] < 0.75 or j[4] < 5 or j[5] < 400:
                    continue
            event_submission.append(j)

    event_submission.sort(key=lambda i: i[3], reverse=True)

    if is_fnd_sort:
        record_list = fnd_sort(event_submission, [3, 4], [3])
        with open('record.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for event in record_list:
                if len(event) == 0:
                    continue
                csvwriter.writerow(event)

        record_list = record_list[:max_submission_num]

        with open(output_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for event in record_list:
                if len(event) == 0:
                    continue
                csvwriter.writerow(event[:3])

            csvfile.close()

    else:
        with open('record.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for event in event_submission:
                if len(event) == 0:
                    continue
                csvwriter.writerow(event)

            csvfile.close()

        event_submission.sort(key=lambda i: i[4], reverse=True)
        event_submission = event_submission[:55000]
        event_submission.sort(key=lambda i: i[3], reverse=True)

        event_submission_limit = event_submission[:max_submission_num]

        with open(output_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for event in event_submission_limit:
                if len(event) == 0:
                    continue
                csvwriter.writerow(event[:3])

            csvfile.close()


if __name__ == '__main__':
    total_begin = datetime.datetime.now()
    config = Config()

    ps_lwinsize = 300
    ps_rwinsize = 100
    is_fnd_sort = True

    min_p_s_time = 25
    max_submission_num = config.cblstm_er_max_submission_num

    use_cpu_rate = config.use_cpu_rate

    data_path = config.re_data_foldername

    model = CBLSTM()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    saver, global_step = Model.continue_previous_session(sess,
                                                         model_file='cblstm',
                                                         ckpt_file='saver/cblstm/checkpoint')

    cblstm_aic('events_re.csv', output_file='submission.csv')

    total_end = datetime.datetime.now()
    print('total time {}'.format(total_end - total_begin))
