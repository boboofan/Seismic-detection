import csv
import obspy
import os
import matplotlib.pyplot as plt
import numpy as np
import pywt

if __name__ == '__main__':
    events_list = []
    with open(r'E:\earthquake\submission\record.csv') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) == 0:
                continue
            row[3] = float(row[3])
            row[4] = float(row[4])
            row[5] = float(row[5])
            # if row[3] <= 0.8 or row[4] <= 7:
            #     continue
            events_list.append(row)
    print(len(events_list))
    events_list = events_list[:40000]
    # events_list.sort(key=lambda i: i[3])
    events_list = events_list[-1::-1]
    # np.random.shuffle(events_list)
    station_dict = {'WDT': 'GS.WDT',
                    'WXT': 'GS.WXT',
                    'LUYA': 'SN.LUYA',
                    'MIAX': 'SN.MIAX',
                    'HSH': 'XX.HSH',
                    'JJS': 'XX.JJS',
                    'JMG': 'XX.JMG',
                    'MXI': 'XX.MXI',
                    'PWU': 'XX.PWU',
                    'QCH': 'XX.QCH',
                    'SPA': 'XX.SPA',
                    'WCH': 'XX.WCH',
                    'XCO': 'XX.XCO',
                    'XJI': 'XX.XJI',
                    'YGD': 'XX.YGD',
                    'YZP': 'XX.YZP'}
    data_file = 'H:\\repecharge\\after'
    total = 0
    for phase in events_list:
        time = obspy.UTCDateTime(phase[1]) - 8 * 3600
        day = str(time.julday)
        station_name = station_dict[phase[0]]
        filename = station_name + '.2008' + day + '000000.'
        bhe = os.path.join(data_file, filename + 'BHE')
        bhn = os.path.join(data_file, filename + 'BHN')
        bhz = os.path.join(data_file, filename + 'BHZ')



        # st_bhe = obspy.read(bhe, starttime=time-60, endtime=time+60)[0].filter('bandpass', freqmin=4, freqmax=15).detrend('spline', order=2, dspline=50).data
        # st_bhn = obspy.read(bhn, starttime=time-60, endtime=time+60)[0].filter('bandpass', freqmin=4, freqmax=15).detrend('spline', order=2, dspline=50).data
        # st_bhz = obspy.read(bhz, starttime=time-60, endtime=time+60)[0].filter('bandpass', freqmin=4, freqmax=15).detrend('spline', order=2, dspline=50).data

        st_bhe = obspy.read(bhe, starttime=time - 60, endtime=time + 60)[0].detrend('spline', order=2, dspline=50).data
        st_bhn = obspy.read(bhn, starttime=time - 60, endtime=time + 60)[0].detrend('spline', order=2, dspline=50).data
        st_bhz = obspy.read(bhz, starttime=time - 60, endtime=time + 60)[0].detrend('spline', order=2, dspline=50).data

        # wt_bhe = pywt.wavedec(st_bhe, 'db2', level=3)
        # wt_bhe = [wt_bhe[0], None, None, None]
        # wt_bhe = pywt.waverec(wt_bhe, 'db2')
        # wt_bhn = pywt.wavedec(st_bhn, 'db2', level=3)
        # wt_bhn = [wt_bhn[0], None, None, None]
        # wt_bhn = pywt.waverec(wt_bhn, 'db2')
        # wt_bhz = pywt.wavedec(st_bhz, 'db2', level=3)
        # wt_bhz = [wt_bhz[0], None, None, None]
        # wt_bhz = pywt.waverec(wt_bhz, 'db2')


        # wt_bhe = pywt.WaveletPacket(st_bhe, 'db2', mode='smooth', maxlevel=3)
        # new_wp = pywt.WaveletPacket(data=None, wavelet='db2', maxlevel=3)
        # new_wp['aaa'] = wt_bhe['aaa'].data
        # new_wp.reconstruct(update=True)
        # wt_bhe = new_wp.data
        # wt_bhn = pywt.WaveletPacket(st_bhn, 'db2', mode='smooth', maxlevel=3)
        # new_wp = pywt.WaveletPacket(data=None, wavelet='db2', maxlevel=3)
        # new_wp['aaa'] = wt_bhn['aaa'].data
        # new_wp.reconstruct(update=True)
        # wt_bhn = new_wp.data
        # wt_bhz = pywt.WaveletPacket(st_bhz, 'db2', mode='smooth', maxlevel=3)
        # new_wp = pywt.WaveletPacket(data=None, wavelet='db2', maxlevel=3)
        # new_wp['aaa'] = wt_bhz['aaa'].data
        # new_wp.reconstruct(update=True)
        # wt_bhz = new_wp.data

        pick_index = 6000
        # wt_index = len(wt_bhe) / 2
        wt_index = pick_index
        # total += 1
        # print(total)
        if phase[2] == 'P':
            color = 'r'
        else:
            color = 'g'

        # print(phase[2], phase[3], phase[4], phase[5])
        print(phase)
        plt.subplot(3, 1, 1)
        plt.plot(st_bhe, color='k')
        plt.axvline(pick_index, color=color)
        plt.axhline(0)
        plt.subplot(3, 1, 2)
        plt.plot(st_bhn, color='k')
        plt.axvline(pick_index, color=color)
        plt.axhline(0)
        plt.subplot(3, 1, 3)
        plt.plot(st_bhz, color='k')
        plt.axvline(pick_index, color=color)
        plt.axhline(0)
        # plt.subplot(6, 1, 2)
        # plt.plot(wt_bhe, color='k')
        # plt.axvline(wt_index, color=color)
        # plt.subplot(6, 1, 4)
        # plt.plot(wt_bhn, color='k')
        # plt.axvline(wt_index, color=color)
        # plt.subplot(6, 1, 6)
        # plt.plot(wt_bhz, color='k')
        # plt.axvline(wt_index, color=color)
        plt.show()


