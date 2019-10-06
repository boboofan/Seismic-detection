# -*- coding: utf-8 -*-

class Config(object):
    def __init__(self):
        self.data_foldername = r'Z:\Users\linjf\preliminary'
        self.re_data_foldername = r'Z:\Users\linjf\repecharge'
        self.file_batch_num = 5
        self.after_per_num = 1
        self.before_per_num = 1
        self.winsize = 6000  # 40s*100
        self.winlag = 1500  # 20s*100

        self.use_cpu_rate = 0.6

        self.cnn_pos_batch_num = 30
        self.cnn_neg_file_batch_num = 10
        self.cnn_neg_file_per_num = 20

        self.p_s_model_pos_batch_num = 10
        self.p_s_model_neg_file_batch_num = 1
        self.p_s_model_neg_file_per_num = 1

        self.birnn_pos_batch_num = self.p_s_model_pos_batch_num
        self.birnn_neg_file_batch_num = self.p_s_model_neg_file_batch_num
        self.birnn_neg_file_per_num = self.p_s_model_neg_file_per_num
        self.birnn_batch_size = self.birnn_pos_batch_num + self.birnn_neg_file_batch_num * self.birnn_neg_file_per_num
        self.LSTM_units_num = 3
        self.birnn_layer_num = 2
        # self.birnn_learning_rate = 1.0
        self.max_grad_norm = 5
        self.keep_prob = 0.7
        # self.lr_decay = 0.8
        self.birnn_class_num = 3

        self.cblstm_pos_batch_num = self.p_s_model_pos_batch_num
        self.cblstm_neg_file_batch_num = self.p_s_model_neg_file_batch_num
        self.cblstm_neg_file_per_num = self.p_s_model_neg_file_per_num
        self.cblstm_batch_size = self.birnn_pos_batch_num + self.birnn_neg_file_batch_num * self.birnn_neg_file_per_num
        # self.cblstm_lstm_units_num = 3
        self.cblstm_lstm_layer_num = 2
        self.cblstm_max_grad_norm = 5
        self.cblstm_keep_prob = 0.65
        self.cblstm_class_num = 3
        self.cblstm_step_size = 100  # 1s * 100

        self.cblstm_er_p_lr_winsize = 50  # 2s * 100
        self.cblstm_er_s_lr_winsize = 50  # 1.5s * 100
        self.cblstm_er_er_p_lr_winsize = 75  # 0.5s * 100
        self.cblstm_er_er_s_lr_winsize = 75  # 0.5s * 100
        self.cblstm_aic_ps_lwinsize = 300
        self.cblstm_aic_ps_rwinsize = 200
        self.cblstm_er_min_p_s_time = 50  # 1s * 100
        self.cblstm_er_preprocess = 'bandpass'
        self.cblstm_er_max_submission_num = 40000

        self.cldnn_step_size = 100
        self.cldnn_pos_batch_num = 10
        self.cldnn_neg_file_batch_num = 2
        self.cldnn_neg_file_per_num = 5
        self.cldnn_keep_prob = 0.5
