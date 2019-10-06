import numpy as np
import csv
import datetime


# def fnd_sort(list_, sort_keys, dom_key, max_outputs=None, reverse=False):
#     """
#     fast-non-dominated-sort
#     if a > b, then a dominates b.
#     :param list_: sort list
#     :param sort_keys: a list contains sort keys
#     :param dom_key: dominated key
#     :param max_outputs:
#     :param reverse: False: if a > b, then return [a, b]
#     :return: a sorted list(contains max_outputs), a sorted list(contains all)
#     """
#     print('fast non dominated sort sort...')
#     list_num = len(list_)
#     if max_outputs is None:
#         max_outputs = list_num
#     sort_keys_len = len(sort_keys)
#
#     f = list()
#     s = list()
#     n = np.zeros([list_num], dtype=int)
#     rank = np.zeros([list_num], dtype=int)
#
#     f.append([])
#     for i, p in enumerate(list_):
#         s.append([])
#         for j, q in enumerate(list_):
#             if j == i:
#                 continue
#             dominated_flag = 1  # 1: p dominates q; -1: q dominates p
#             for key in sort_keys:
#                 if p[key] < q[key]:
#                     dominated_flag = 0
#                     break
#             if dominated_flag == 0:
#                 dominated_flag = -1
#                 for key in sort_keys:
#                     if p[key] > q[key]:
#                         dominated_flag = 0
#                         break
#
#             if dominated_flag == 1:
#                 s[i].append(j)
#             if dominated_flag == -1:
#                 n[i] += 1
#         if n[i] == 0:
#             rank[i] = 1
#             f[0].append(i)
#         if i % 10 == 0:
#             print('num {} done.'.format(i))
#
#     i = 0
#     while len(f[i]) != 0:
#         Q = list()
#         for p in f[i]:
#             for q in s[p]:
#                 n[q] -= 1
#                 if n[q] == 0:
#                     rank[q] = i + 1
#                     Q.append(q)
#         i += 1
#         f.append(Q)
#         if i % 100 == 0:
#             print('f {} done.'.format(i))
#
#     current_output_num = 0
#     sorted_list_limit = []
#     for index_list in f:
#         if len(index_list) + current_output_num <= max_outputs:
#             for ind in index_list:
#                 sorted_list_limit.append(list_[ind])
#                 current_output_num += 1
#         else:
#             append_num = max_outputs - current_output_num
#             current_list = [list_[i] for i in index_list]
#             current_list.sort(key=lambda i: i[dom_key], reverse=not reverse)
#             i = 0
#             while i < append_num:
#                 sorted_list_limit.append(current_list[i])
#                 i += 1
#     sorted_list = list()
#     for index_list in f:
#         tmp_list = [list_[i] for i in index_list]
#         tmp_list.sort(key=lambda i: i[dom_key], reverse=not reverse)
#         sorted_list = sorted_list + tmp_list
#
#     return sorted_list_limit, sorted_list

def fnd_sort(list_, sort_keys, dom_key, reverse=False):
    """
    fast-non-dominated-sort
    if a > b, then a dominates b.
    :param list_: sort list
    :param sort_keys: a list contains sort keys
    :param dom_key: dominated key
    :param reverse: False: if a > b, then return [a, b]
    :return: a sorted list
    """
    print('fast non dominated sort sort...')
    list_num = len(list_)

    f = list()
    s = np.zeros([list_num, list_num], dtype=np.int32)
    s_list_len = np.zeros([list_num], dtype=int)
    n = np.zeros([list_num], dtype=int)
    rank = np.zeros([list_num], dtype=int)

    f.append([])
    print('total num {}.'.format(list_num))
    begin = datetime.datetime.now()
    for i, p in enumerate(list_):
        s_list_len[i] = 0
        for j, q in enumerate(list_):
            if j == i:
                continue
            dominated_flag = 1  # 1: p dominates q; -1: q dominates p
            for key in sort_keys:
                if p[key] < q[key]:
                    dominated_flag = 0
                    break
            if dominated_flag == 0:
                dominated_flag = -1
                for key in sort_keys:
                    if p[key] > q[key]:
                        dominated_flag = 0
                        break

            if dominated_flag == 1:
                s[i, s_list_len[i]] = j
                s_list_len[i] += 1
            if dominated_flag == -1:
                n[i] += 1
        if n[i] == 0:
            rank[i] = 1
            f[0].append(i)
        if i % 1000 == 0:
            end = datetime.datetime.now()
            print('num {} done, time {}.'.format(i, end - begin))
            begin = datetime.datetime.now()

    i = 0
    begin = datetime.datetime.now()
    while len(f[i]) != 0:
        Q = list()
        for p in f[i]:
            for p_s_index in range(s_list_len[p]):
                q = s[p, p_s_index]
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        f.append(Q)
        if i % 50 == 0:
            end = datetime.datetime.now()
            print('f {} done, time {}.'.format(i, end - begin))
            begin = datetime.datetime.now()

    sorted_list = list()
    for index_list in f:
        tmp_list = [list_[i] for i in index_list]
        for key_index in dom_key:
            tmp_list.sort(key=lambda i: i[key_index], reverse=not reverse)
        sorted_list = sorted_list + tmp_list

    return sorted_list


if __name__ == '__main__':
    submission_list = []
    with open(r'record_raw.csv') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) == 0:
                continue
            row[3] = float(row[3])
            row[4] = float(row[4])
            if row[3] < 0.75 or row[4] < 5:
                continue
            submission_list.append(row)
        file.close()


    record_list = fnd_sort(submission_list, [3, 4], [3])
    submission_list = record_list[:40000]

    with open('record.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for event in record_list:
            if len(event) == 0:
                continue
            csvwriter.writerow(event)

        csvfile.close()

    with open('submission.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for event in submission_list:
            if len(event) == 0:
                continue
            csvwriter.writerow(event[:3])

        csvfile.close()
