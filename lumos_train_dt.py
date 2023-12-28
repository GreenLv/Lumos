#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021.05.24
# @Author  : Gerui Lv
# @Version : $v3.9.1$

import warnings
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import make_scorer, r2_score
from sklearn.metrics import mean_squared_error as mse
from io import StringIO
import pydotplus
from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import sys
import csv
import os

"""Read data and train a decision tree (including serching for the best params)

Run beyond 'data_output/'
"""

# settings
TARGET_TPYE = 5  # 0.time, 1.throughput, 2.1/throughput, 3.lg_time, 4.time^2, 5.lg_throughput
PAST_INFO_LEN = 5    # number of past chunks info that are logged
HM_LEN = 5
STATE_LEN = 1        # number of past chunks info that are really used
MUTUAL_INFO = False   # mutual information between features and labels
FILTER_OUTLIER = False  # whether to filter outliers (OUTLIER_TRESH) or not

MLR = True           # multiple linear regressor

REG_TREE = True      # DecisionTreeRegressor
REG_PRUNE = True     # prepruning and postpruning for reg
REG_TUNE = True      # tune for prepruning and postpruning for reg

CLF_TREE = False     # DecisionTreeClassifier
CLF_PRUNE = True     # prepruning and postpruning for clf
CLF_TUNE = True      # tune for prepruning and postpruning for clf

EXPORT_RES = True    # save info, plot figure and export tree for python and js (TREE_DIR)
PLOT_DEPTH = 5       # depth of tree in .png

if TARGET_TPYE == 0:
    CLASS_POINT = [0, 0.45, 0.83, 1.37, 2.32, 4.0, 8.0, 12.0]  # s
elif TARGET_TPYE == 1:
    CLASS_POINT = [0, 1.7, 2.7, 4.9, 6.8, 9.8, 19.3]  # Mbps
elif TARGET_TPYE == 2:
    CLASS_POINT = [0, 0.05, 0.1, 0.15, 0.20, 0.35, 0.6]  # s/Mb
elif TARGET_TPYE == 3:
    CLASS_POINT = [-3, -0.386, -0.089, 0.152, 0.385, 1]
elif TARGET_TPYE == 4:
    CLASS_POINT = [0, 0.2025, 0.6889, 1.8769, 5.3824, 16, 64, 144]  # s^2
elif TARGET_TPYE == 5:
    # CLASS_POINT = [-3, 0.632, 0.8, 0.935, 1.199]
    CLASS_POINT = [0.0, 2.408, 4.287, 6.307, 8.615, 15.811]

# decision tree params for regressor
R_CRITERION_PARAM = ['mse', 'friedman_mse', 'mae', 'poisson']   # for regressor
R_MAX_LEAF_NODES_PARAM = [None, *range(40, 101, 10), 150, 200]
R_MAX_DEPTH_PARAM = [None, *range(5, 19, 1)]
# R_MIN_SAMPLES_LEAF_PARAM = [1, 25, 50, 75, 100, 200]
R_MIN_SAMPLES_LEAF_PARAM = [1, 5, 10, 20, 25, 30, 40, 50]  # strong: 0, 1
R_MIN_IMPURITY_PARAM = [*np.linspace(0, 0.5, 5)]
R_CCP_ALPHA_PARAM = [1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 1.25e-4, 1.5e-4]
# R_CCP_ALPHA_PARAM = [0.01, 0.025, 0.05, 0.075, 0.1] # 1

# decision tree params for classifier
GINI_TRHESHOLD = 0.5
ENTROPY_TRHESHOLD = 1
CRITERION_PARAM = ['gini', 'entropy']
MAX_LEAF_NODES_PARAM = [None, 50, 100, 250, 500, 750, 1000]
MAX_DEPTH_PARAM = [None, 5, 7, 10, 20, 30, 40, 50, 75, 100]
MIN_SAMPLES_LEAF_PARAM = [1, 1e-5, 5e-5, 1e-4, 5e-4, *np.linspace(0.001, 0.01, 5)]
MIN_IMPURITY_PARAM = [*np.linspace(0, GINI_TRHESHOLD, 20)]
CCP_ALPHA_PARAM = [1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 1.25e-4, 1.5e-4]

# best params for regressor
R_CRITERION = 'mse'
R_MAX_DEPTH = 9
R_MAX_LEAF_NODES = None
R_MIN_SAMPLES_LEAF = 10
R_MIN_IMPURITY_DECREASE = 0.0
R_CCP_ALPHA = 7.5e-5

# best params for classifier
CRITERION = 'entropy'
MAX_LEAF_NODES = 250
MAX_DEPTH = None
MIN_SAMPLES_LEAF = 1
MIN_IMPURITY_DECREASE = 0.0
CCP_ALPHA = 0.000125

# dir path
DATA_DIR = './data_output/'
TREE_DIR = './tree/'

# settings
RANDOM_SEED = 42
K = 5   # k-fold cross validation
TRAIN_FRAC = 0.75
CPP_THRESHOLD = 300
OUTLIER_TRESH = 100  # Mbps

# constant value
BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
M_IN_K = 1000.0
VIDEO_SOURCE = ['bbb', 'bbb-an', 'bbb-4s-an', 'ed-an', 'ed-4s-an']
DB = ['50M', '5M']
CT = ['wifi', '4g']
SS = ['strong', 'medium', 'weak']
PS = ['buffering', 'steady']

# FEATURE_NAMES = ['downstream_bandwidth', 'link_stat', *['connection_type_' + ct for ct in CT],
# 'signal_strength', 'bitrate', 'chunk_size']
FEATURE_NAMES = ['Max throughput', 'Max delivery time', 'Connection Type',
                 'Target Bitrate', 'Target chunk size']
TREE_NAMES = ['Unpruned', 'Prepruning', 'Postpruning']
DEFAULT_FONT_FAMILY = 'Times New Roman'
# DEFAULT_FONT_FAMILY = 'Sans Serif'
BAR_COLOR = '#15AD67'


def mkdir(dir):
    is_exists = os.path.exists(dir)
    if not is_exists:
        os.makedirs(dir)


def lower_bound(arr, target, start):
    """binary search from the start for the first number
    smaller than or equal to the target, return the index
    """

    low = start
    high = len(arr) - 1

    if target < arr[low]:
        return -1
    elif target >= arr[high]:
        return high

    mid = start
    last_mid = None

    while last_mid != mid:
        last_mid = mid
        mid = int((low + high) / 2)
        if arr[mid] <= target:
            low = mid
        else:
            high = mid

    return mid


def get_data_and_label(data_dir):
    print('-' * 50)
    print('Reading data...')
    if CLF_TREE:
        print('CLASS_POINT: ' + str(CLASS_POINT))

    raw_states = []
    raw_class_labels = []
    raw_value_labels = []

    files = os.listdir(data_dir)
    for file in files:
        if file.endswith('.csv'):
            csv_path = data_dir + file

            test_date = file[:-4].split('_')[0]
            test_name = test_date + '/' + file[:-4].split('_')[1] + '/'
            video = file[:-4].split('_')[-2]
            assert(video in VIDEO_SOURCE or video == 'ed'), \
                '[get_data_and_label] Error! Video does not exist: %s' % (video)

            with open(csv_path, 'r') as csv_data:
                next(csv_data)
                reader = csv.reader(csv_data)

                past_bitrate = [0] * PAST_INFO_LEN
                past_chunk_size = [0] * PAST_INFO_LEN
                past_delivery_time = [0] * PAST_INFO_LEN
                past_player_state = [0] * PAST_INFO_LEN
                past_chunk_index = [0] * PAST_INFO_LEN
                past_app_throughput = [0] * PAST_INFO_LEN

                for row in reader:
                    app_throughput = int(row[5]) / M_IN_K
                    if FILTER_OUTLIER and app_throughput >= OUTLIER_TRESH:
                        continue
                    delivery_time = float(row[6])

                    # label
                    if TARGET_TPYE == 0:
                        label_class = lower_bound(CLASS_POINT, delivery_time, 0)
                        label_value = delivery_time
                    elif TARGET_TPYE == 1:
                        label_class = lower_bound(CLASS_POINT, app_throughput, 0)
                        label_value = app_throughput
                    elif TARGET_TPYE == 2:
                        label_value = M_IN_K / int(row[5])
                        label_class = lower_bound(CLASS_POINT, label_value, 0)
                    elif TARGET_TPYE == 3:
                        label_value = math.log(delivery_time, 10)
                        label_class = lower_bound(CLASS_POINT, label_value, 0)
                    elif TARGET_TPYE == 4:
                        label_value = delivery_time ** 2
                        label_class = lower_bound(CLASS_POINT, label_value, 0)
                    elif TARGET_TPYE == 5:
                        label_value = math.log(app_throughput, 10)
                        label_class = lower_bound(CLASS_POINT, app_throughput, 0)
                    # state (input)
                    state = []

                    # a. network stat
                    downstream_bandwidth = np.max(past_app_throughput)
                    link_stat = max(past_delivery_time)
                    connection_type = CT.index(row[1]) / (len(CT) - 1)
                    # connection_type = [0] * len(CT)
                    # connection_type[CT.index(row[1])] = 1  # one-hot
                    signal_strength = SS.index(row[2]) / (len(SS) - 1)
                    state.append(downstream_bandwidth)
                    state.append(link_stat)
                    state.append(connection_type)
                    # state.extend(connection_type)
                    # state.append(signal_strength)

                    # b. next chunk info
                    bitrate = float(row[3]) / M_IN_K
                    chunk_size = int(row[4]) / M_IN_K
                    state.append(bitrate)
                    state.append(chunk_size)

                    # c. past chunk info
                    state.extend(past_bitrate[-STATE_LEN:])
                    state.extend(past_chunk_size[-STATE_LEN:])
                    if TARGET_TPYE in [0, 3, 4]:
                        state.extend(past_delivery_time[-STATE_LEN:])
                    if TARGET_TPYE in [1, 2, 5]:
                        state.extend(past_app_throughput[-STATE_LEN:])
                    # state.extend(past_player_state[-STATE_LEN:])
                    state.extend(past_chunk_index[-STATE_LEN:])

                    player_state = PS.index(row[7]) / (len(PS) - 1)
                    chunk_index = int(row[8]) / 10.0

                    past_bitrate.append(bitrate)
                    past_chunk_size.append(chunk_size)
                    past_delivery_time.append(delivery_time)
                    past_player_state.append(player_state)
                    past_chunk_index.append(chunk_index)
                    past_app_throughput.append(app_throughput)

                    past_bitrate.pop(0)
                    past_chunk_size.pop(0)
                    past_delivery_time.pop(0)
                    past_player_state.pop(0)
                    past_chunk_index.pop(0)
                    past_app_throughput.pop(0)

                    raw_states.append(state)
                    raw_class_labels.append(label_class)
                    raw_value_labels.append(label_value)

                    # debug
                    # print(state)
                    # print('label: ' + str(label_value) + ', ' + str(label_class))

                    # if i >= TIME_SLOT:
                    #     break
                    # i = i + 1

        # break

    print('Data ready.')
    # print(raw_states[:15])
    # print(raw_class_labels[:15])
    # print(raw_value_labels[:15])
    return raw_states, raw_class_labels, raw_value_labels


def set_feature_names(features):
    global FEATURE_NAMES
    assert(STATE_LEN >= 1 and STATE_LEN <= PAST_INFO_LEN), '[set_feature_names] STATE_LEN error!'
    if STATE_LEN == 1:
        FEATURE_NAMES.extend(['Past bitrate', 'Past chunk size'])
        # FEATURE_NAMES.extend(['Past chunk size'])
        if TARGET_TPYE in [0, 3, 4]:
            FEATURE_NAMES.append('Past delivery time',)
        if TARGET_TPYE in [1, 2, 5]:
            FEATURE_NAMES.append('Past throughput',)
        # FEATURE_NAMES.extend(['Past player\'s state', 'Past chunk index'])
        FEATURE_NAMES.extend(['Past chunk index'])
    else:
        FEATURE_NAMES.extend(['Past bitrate ' + str(i) for i in range(STATE_LEN, 0, -1)])
        FEATURE_NAMES.extend(['Past chunk size ' + str(i) for i in range(STATE_LEN, 0, -1)])
        if TARGET_TPYE in [0, 3, 4]:
            FEATURE_NAMES.extend(['Past delivery time ' + str(i) for i in range(STATE_LEN, 0, -1)])
        if TARGET_TPYE in [1, 2, 5]:
            FEATURE_NAMES.extend(['Past throughput ' + str(i) for i in range(STATE_LEN, 0, -1)])
        # FEATURE_NAMES.extend(['Past player\'s state' + str(i) for i in range(STATE_LEN, 0, -1)])
        FEATURE_NAMES.extend(['Past chunk index' + str(i) for i in range(STATE_LEN, 0, -1)])

    assert(len(FEATURE_NAMES) == len(features[0]))
    print('Features: ' + str(FEATURE_NAMES))


def feature_mutual_info(features, labels, values):
    print('-' * 50)
    print('Calculating mutual information...')
    # mic = mutual_info_classif(features, labels)
    # print('Mutual information of features: ')
    # for feature, mi in zip(FEATURE_NAMES, mic):
    #     print('{}: {:.3f}'.format(feature, mi))

    mir = mutual_info_regression(features, values)
    print('Mutual information of features: ')
    for feature, mi in zip(FEATURE_NAMES, mir):
        print('{}: {:.3f}'.format(feature, mi))

    return mir


def class_cnt(train_labels, val_labels):
    print('-' * 50)

    train_feq_list = {}
    train_cnt = {}
    val_cnt = {}
    classes = range(len(CLASS_POINT))

    for i in classes:
        train_feq = train_labels.count(i) / len(train_labels)
        train_feq_list[i] = train_feq
        train_cnt[i] = '{:.2%}'.format(train_feq)
        val_feq = val_labels.count(i) / len(val_labels)
        val_cnt[i] = '{:.2%}'.format(val_feq)

    print('Training data count: ' + str(train_cnt))
    print('Validating data count: ' + str(val_cnt))

    return classes, train_cnt, val_cnt


def train_mlr(states, values):
    print('-' * 50)

    train_states, val_states, train_values, val_values = \
        train_test_split(states, values, train_size=TRAIN_FRAC, random_state=RANDOM_SEED)

    mlr = linear_model.LinearRegression()
    mlr.fit(train_states, train_values)
    print('Multiple linear regressor is traind.')

    regr_t_cod = r2_score(train_values, mlr.predict(train_states))
    regr_v_cod = r2_score(val_values, mlr.predict(val_states))
    print('MLR training coefficient of determination: {:.3f}'.format(regr_t_cod))
    print('MLR validating coefficient of determination: {:.3f}'.format(regr_v_cod))
    regr_t_loss = mse(train_values, mlr.predict(train_states))
    regr_v_loss = mse(val_values, mlr.predict(val_states))
    print('MLR training loss: {:.3f}'.format(regr_t_loss))
    print('MLR validating loss: {:.3f}'.format(regr_v_loss))

    mlr_metrics = [regr_t_cod, regr_v_cod, regr_t_loss, regr_v_loss]

    coefficients = [float('{:.4f}'.format(coef)) for coef in mlr.coef_]
    intercept = '{:.4f}'.format(mlr.intercept_)
    print('MLR coefficients: ' + str(coefficients))
    print('MLR intercept: ', intercept)

    mlr_params = [coefficients, intercept]

    # cross validation
    scoring = {'r2', 'neg_mean_squared_error'}
    scores = cross_validate(mlr, states, values, scoring=scoring, cv=K, return_train_score=True)
    # items = sorted(scores.keys())
    mlr_cv = [np.mean(scores['train_r2']),
              np.mean(scores['test_r2']),
              -1 * np.mean(scores['train_neg_mean_squared_error']),
              -1 * np.mean(scores['test_neg_mean_squared_error'])]

    print('Cross validation performance: ')
    print('CV training coefficient of determination: {:.3f}'.format(mlr_cv[0]))
    print('CV validating coefficient of determination: {:.3f}'.format(mlr_cv[1]))
    print('CV training loss: {:.3f}'.format(mlr_cv[2]))
    print('CV validating loss: {:.3f}'.format(mlr_cv[3]))
    return mlr, mlr_metrics, mlr_params, mlr_cv


def train_unpruned_reg(states, values):
    print('-' * 50)

    train_states, val_states, train_values, val_values = \
        train_test_split(states, values, train_size=TRAIN_FRAC, random_state=RANDOM_SEED)

    reg = tree.DecisionTreeRegressor(random_state=RANDOM_SEED)
    reg.fit(train_states, train_values)
    print('Regressor is traind.')

    reg_t_loss = mse(train_values, reg.predict(train_states))
    reg_v_loss = mse(val_values, reg.predict(val_states))
    print('Unpruned regressor training loss: {:.3f}'.format(reg_t_loss))
    print('Unpruned regressor validating loss: {:.3f}'.format(reg_v_loss))

    reg_metrics = [reg_t_loss, reg_v_loss]

    # cross validation
    scoring = {'neg_mean_squared_error'}
    scores = cross_validate(reg, states, values, scoring=scoring, cv=K, return_train_score=True)
    items = sorted(scores.keys())
    reg_cv = [-1 * np.mean(scores['train_neg_mean_squared_error']),
              -1 * np.mean(scores['test_neg_mean_squared_error'])]

    print('Cross validation performance: ')
    print('CV training loss: {:.3f}'.format(reg_cv[0]))
    print('CV validating loss: {:.3f}'.format(reg_cv[1]))

    return reg, reg_metrics, reg_cv


def tune_for_reg_prepruning(states, values):
    print('Searching the best parameters for reg prepruning...')

    # all params
    # parameters = {'criterion': R_CRITERION_PARAM,
    #               'max_depth': R_MAX_DEPTH_PARAM,
    #               'max_leaf_nodes': R_MAX_LEAF_NODES_PARAM,
    #               'min_samples_leaf': R_MIN_SAMPLES_LEAF_PARAM,
    #               'min_impurity_decrease': R_MIN_IMPURITY_PARAM}
    # useful params
    parameters = {'max_depth': R_MAX_DEPTH_PARAM,
                  'min_samples_leaf': R_MIN_SAMPLES_LEAF_PARAM}
    scoring = {'neg_mean_squared_error'}

    reg = tree.DecisionTreeRegressor(random_state=RANDOM_SEED)
    gs = GridSearchCV(reg, param_grid=parameters, scoring=scoring, n_jobs=-1, cv=K,
                      return_train_score=True, refit='neg_mean_squared_error')

    train_states, val_states, train_values, val_values = \
        train_test_split(states, values, train_size=TRAIN_FRAC, random_state=RANDOM_SEED)
    gs.fit(train_states, train_values)

    scores = gs.cv_results_
    index = gs.best_index_
    best_perf = [-1 * scores['mean_train_neg_mean_squared_error'][index],
                 -1 * scores['mean_test_neg_mean_squared_error'][index]]
    bset_param = gs.best_params_

    print('Searching ends. Bset params: {}'.format(bset_param))

    return gs.best_estimator_, bset_param, best_perf


def reg_prepruning(states, values):
    print('-' * 50)
    print('Regressor prepruning...')

    train_states, val_states, train_values, val_values = \
        train_test_split(states, values, train_size=TRAIN_FRAC, random_state=RANDOM_SEED)

    if REG_TUNE:
        pre_reg, pre_reg_params, pre_reg_cv = tune_for_reg_prepruning(states, values)
    else:
        # best param
        pre_reg = tree.DecisionTreeRegressor(criterion=R_CRITERION,
                                             max_depth=R_MAX_DEPTH,
                                             max_leaf_nodes=R_MAX_LEAF_NODES,
                                             min_samples_leaf=R_MIN_SAMPLES_LEAF,
                                             min_impurity_decrease=R_MIN_IMPURITY_DECREASE,
                                             random_state=RANDOM_SEED)
        pre_reg_params = {'criterion': R_CRITERION,
                          'max_depth': R_MAX_DEPTH,
                          'max_leaf_nodes': R_MAX_LEAF_NODES,
                          'min_samples_leaf': R_MIN_SAMPLES_LEAF,
                          'min_impurity_decrease': R_MIN_IMPURITY_DECREASE}
        pre_reg.fit(train_states, train_values)

    assert(pre_reg), '[reg_prepruning] Error! There is no pre_reg.'
    print('Prepruning done.')

    pre_reg_t_loss = mse(train_values, pre_reg.predict(train_states))
    pre_reg_v_loss = mse(val_values, pre_reg.predict(val_states))
    print('Prepruning regressor training loss: {:.3f}'.format(pre_reg_t_loss))
    print('Prepruning regressor validating loss: {:.3f}'.format(pre_reg_v_loss))

    pre_reg_metrics = [pre_reg_t_loss, pre_reg_v_loss]

    # cross validation
    if not REG_TUNE:
        scoring = {'neg_mean_squared_error'}
        scores = cross_validate(pre_reg, states, values, scoring=scoring,
                                cv=K, return_train_score=True)
        # print(sorted(scores.keys())
        pre_reg_cv = [-1 * np.mean(scores['train_neg_mean_squared_error']),
                      -1 * np.mean(scores['test_neg_mean_squared_error'])]

    print('Cross validation performance: ')
    print('CV training loss: {:.3f}'.format(pre_reg_cv[0]))
    print('CV validating loss: {:.3f}'.format(pre_reg_cv[1]))

    return pre_reg, pre_reg_metrics, pre_reg_params, pre_reg_cv


def tune_for_reg_postpruning(states, values, ccp_alphas):
    print('Searching the best ccp_alpha for reg postpruning...')

    parameters = {'ccp_alpha': ccp_alphas}
    scoring = {'neg_mean_squared_error'}

    reg = tree.DecisionTreeRegressor(random_state=RANDOM_SEED)
    gs = GridSearchCV(reg, param_grid=parameters, scoring=scoring, n_jobs=-1, cv=K,
                      return_train_score=True, refit='neg_mean_squared_error')

    train_states, val_states, train_values, val_values = \
        train_test_split(states, values, train_size=TRAIN_FRAC, random_state=RANDOM_SEED)
    gs.fit(train_states, train_values)

    scores = gs.cv_results_
    index = gs.best_index_
    # print(scores.keys())
    best_perf = [-1 * scores['mean_train_neg_mean_squared_error'][index],
                 -1 * scores['mean_test_neg_mean_squared_error'][index]]
    bset_param = gs.best_params_
    print('Searching ends. Bset params: {}'.format(bset_param))

    return gs.best_estimator_, bset_param, best_perf


def reg_postpruning(states, values):
    print('-' * 50)
    print('Regressor postpruning...')

    train_states, val_states, train_values, val_values = \
        train_test_split(states, values, train_size=TRAIN_FRAC, random_state=RANDOM_SEED)

    if REG_TUNE:
        reg = tree.DecisionTreeRegressor(random_state=RANDOM_SEED)
        path = reg.cost_complexity_pruning_path(train_states, train_values)
        raw_ccp_alphas, _ = path.ccp_alphas, path.impurities
        start = lower_bound(raw_ccp_alphas, R_CCP_ALPHA_PARAM[0], 0)
        end = lower_bound(raw_ccp_alphas, R_CCP_ALPHA_PARAM[-1], 0)
        ccp_alphas = raw_ccp_alphas[start:end]
        if len(ccp_alphas) >= CPP_THRESHOLD:    # too many params
            ccp_alphas = R_CCP_ALPHA_PARAM
            # ccp_alphas = np.linspace(np.min(R_CCP_ALPHA_PARAM),
            #                          np.max(R_CCP_ALPHA_PARAM), CPP_THRESHOLD)
        post_reg, post_reg_param, post_reg_cv = \
            tune_for_reg_postpruning(states, values, ccp_alphas)
    else:
        post_reg = tree.DecisionTreeRegressor(ccp_alpha=R_CCP_ALPHA, random_state=RANDOM_SEED)
        post_reg_param = {'ccp_alpha': R_CCP_ALPHA}
        post_reg.fit(train_states, train_values)

    assert(post_reg), '[reg_postpruning] Error! There is no post_reg.'
    print('Postpruning done.')

    post_reg_t_loss = mse(train_values, post_reg.predict(train_states))
    post_reg_v_loss = mse(val_values, post_reg.predict(val_states))
    print('Postpruning regressor training loss: {:.3f}'.format(post_reg_t_loss))
    print('Postpruning regressor validating loss: {:.3f}'.format(post_reg_v_loss))

    post_reg_metrics = [post_reg_t_loss, post_reg_v_loss]

    # cross validation
    if not REG_TUNE:
        scoring = {'neg_mean_squared_error'}
        scores = cross_validate(post_reg, states, values, scoring=scoring,
                                cv=K, return_train_score=True)
        # items = sorted(scores.keys())
        post_reg_cv = [-1 * np.mean(scores['train_neg_mean_squared_error']),
                       -1 * np.mean(scores['test_neg_mean_squared_error'])]

    print('Cross validation performance: ')
    print('CV training loss: {:.3f}'.format(post_reg_cv[0]))
    print('CV validating loss: {:.3f}'.format(post_reg_cv[1]))

    return post_reg, post_reg_metrics, post_reg_param, post_reg_cv


def train_regressor(states, values):
    reg, reg_metrics, reg_cv = train_unpruned_reg(states, values)

    if REG_PRUNE:
        # ----------prepruning (select best params)----------
        pre_reg, pre_reg_metrics, pre_reg_params, pre_reg_cv = reg_prepruning(states, values)

        # ----------postpruning by CPP----------
        post_reg, post_reg_metrics, post_reg_param, post_reg_cv = reg_postpruning(states, values)

        # ----------select the best regressor according to loss on validating set----------
        all_reg = [reg, pre_reg, post_reg]
        all_reg_metrics = [reg_metrics, pre_reg_metrics, post_reg_metrics]
        all_params = [None, pre_reg_params, post_reg_param]
        all_cv = [reg_cv, pre_reg_cv, post_reg_cv]
        all_loss = [metric[1] for i, metric in enumerate(all_cv)]
        min_loss = min(all_loss)

        best_index = all_loss.index(min_loss)
        best_reg = all_reg[best_index]
        best_metrics = all_reg_metrics[best_index]
        bset_params = all_params[best_index]
        best_cv = all_cv[best_index]
        print('-' * 50)
        print('Select best regressor: %s' % (TREE_NAMES[best_index]))
        return best_reg, best_metrics, best_cv, best_index, bset_params

    return reg, reg_metrics, reg_cv, None, None


def train_unpruned_clf(states, labels):
    train_states, val_states, train_labels, val_labels = \
        train_test_split(states, labels, train_size=TRAIN_FRAC, random_state=RANDOM_SEED)

    _, train_cnt, val_cnt = class_cnt(train_labels, val_labels)

    print('-' * 50)

    clf = tree.DecisionTreeClassifier(random_state=RANDOM_SEED)
    clf.fit(train_states, train_labels)
    print('Classifier is traind.')

    # clf_t_precision = np.mean(clf.predict(train_states) == train_labels)
    # clf_v_precision = np.mean(clf.predict(val_states) == val_labels)
    clf_t_precision = clf.score(train_states, train_labels)
    clf_v_precision = clf.score(val_states, val_labels)
    print('Unpruned classifier training precision: {:.3%}'.format(clf_t_precision))
    print('Unpruned classifier validating precision: {:.3%}'.format(clf_v_precision))
    clf_t_loss = mse(train_labels, clf.predict(train_states))
    clf_v_loss = mse(val_labels, clf.predict(val_states))
    print('Unpruned classifier training loss: {:.3f}'.format(clf_t_loss))
    print('Unpruned classifier validating loss: {:.3f}'.format(clf_v_loss))

    clf_metrics = [clf_t_precision, clf_v_precision, clf_t_loss, clf_v_loss]

    # cross validation
    scoring = {'accuracy', 'neg_mean_squared_error'}
    scores = cross_validate(clf, states, labels, scoring=scoring, cv=K, return_train_score=True)
    # items = sorted(scores.keys())
    clf_cv = [np.mean(scores['train_accuracy']),
              np.mean(scores['test_accuracy']),
              -1 * np.mean(scores['train_neg_mean_squared_error']),
              -1 * np.mean(scores['test_neg_mean_squared_error'])]

    print('Cross validation performance: ')
    print('CV training precision: {:.3%}'.format(clf_cv[0]))
    print('CV validating precision: {:.3%}'.format(clf_cv[1]))
    print('CV training loss: {:.3f}'.format(clf_cv[2]))
    print('CV validating loss: {:.3f}'.format(clf_cv[3]))

    return clf, clf_metrics, clf_cv, train_cnt, val_cnt


def tune_for_clf_prepruning(states, labels):
    print('Searching the best parameters for clf prepruning...')

    # all params
    # parameters = {'criterion': CRITERION_PARAM,
    #               'max_depth': MAX_DEPTH_PARAM,
    #               'max_leaf_nodes': MAX_LEAF_NODES_PARAM,
    #               'min_samples_leaf': MIN_SAMPLES_LEAF_PARAM,
    #               'min_impurity_decrease': MIN_IMPURITY_PARAM}
    # useful params
    parameters = {'criterion': CRITERION_PARAM,
                  'max_depth': MAX_DEPTH_PARAM,
                  'min_samples_leaf': MIN_SAMPLES_LEAF_PARAM}
    # parameters = {'max_depth': [5, 10]}
    scoring = {'accuracy', 'neg_mean_squared_error'}

    clf = tree.DecisionTreeClassifier(random_state=RANDOM_SEED)
    gs = GridSearchCV(clf, param_grid=parameters, scoring=scoring, n_jobs=-1, cv=K,
                      return_train_score=True, refit='neg_mean_squared_error')

    train_states, val_states, train_labels, val_labels = \
        train_test_split(states, labels, train_size=TRAIN_FRAC, random_state=RANDOM_SEED)
    gs.fit(train_states, train_labels)

    scores = gs.cv_results_
    index = gs.best_index_
    best_perf = [scores['mean_train_accuracy'][index],
                 scores['mean_test_accuracy'][index],
                 -1 * scores['mean_train_neg_mean_squared_error'][index],
                 -1 * scores['mean_test_neg_mean_squared_error'][index]]
    bset_param = gs.best_params_

    print('Searching ends. Bset params: {}'.format(bset_param))

    return gs.best_estimator_, bset_param, best_perf


def clf_prepruning(states, labels):
    print('-' * 50)
    print('Classifier prepruning...')

    train_states, val_states, train_labels, val_labels = \
        train_test_split(states, labels, train_size=TRAIN_FRAC, random_state=RANDOM_SEED)

    if CLF_TUNE:
        pre_clf, pre_clf_params, pre_clf_cv = tune_for_clf_prepruning(states, labels)
    else:
        # best param
        pre_clf = tree.DecisionTreeClassifier(criterion=CRITERION,
                                              max_depth=MAX_DEPTH,
                                              max_leaf_nodes=MAX_LEAF_NODES,
                                              min_samples_leaf=MIN_SAMPLES_LEAF,
                                              min_impurity_decrease=MIN_IMPURITY_DECREASE,
                                              random_state=RANDOM_SEED)
        pre_clf_params = {'criterion': CRITERION,
                          'max_depth': MAX_DEPTH,
                          'max_leaf_nodes': MAX_LEAF_NODES_PARAM,
                          'min_samples_leaf': MIN_SAMPLES_LEAF,
                          'min_impurity_decrease': MIN_IMPURITY_DECREASE}
        pre_clf.fit(train_states, train_labels)

    assert(pre_clf), '[clf_prepruning] Error! There is no pre_clf.'
    print('Prepruning done.')

    pre_clf_t_precision = pre_clf.score(train_states, train_labels)
    pre_clf_v_precision = pre_clf.score(val_states, val_labels)
    print('Prepruning classifier training precision: {:.3%}'.format(pre_clf_t_precision))
    print('Prepruning classifier validating precision: {:.3%}'.format(pre_clf_v_precision))
    pre_clf_t_loss = mse(train_labels, pre_clf.predict(train_states))
    pre_clf_v_loss = mse(val_labels, pre_clf.predict(val_states))
    print('Prepruning classifier training loss: {:.3f}'.format(pre_clf_t_loss))
    print('Prepruning classifier validating loss: {:.3f}'.format(pre_clf_v_loss))

    pre_clf_metrics = [pre_clf_t_precision, pre_clf_v_precision, pre_clf_t_loss, pre_clf_v_loss]

    # cross validation
    if not CLF_TUNE:
        scoring = {'accuracy', 'neg_mean_squared_error'}
        scores = cross_validate(pre_clf, states, labels, scoring=scoring,
                                cv=K, return_train_score=True)
        # print(sorted(scores.keys())
        pre_clf_cv = [np.mean(scores['train_accuracy']),
                      np.mean(scores['test_accuracy']),
                      -1 * np.mean(scores['train_neg_mean_squared_error']),
                      -1 * np.mean(scores['test_neg_mean_squared_error'])]

    print('Cross validation performance: ')
    print('CV training precision: {:.3%}'.format(pre_clf_cv[0]))
    print('CV validating precision: {:.3%}'.format(pre_clf_cv[1]))
    print('CV training loss: {:.3f}'.format(pre_clf_cv[2]))
    print('CV validating loss: {:.3f}'.format(pre_clf_cv[3]))

    return pre_clf, pre_clf_metrics, pre_clf_params, pre_clf_cv


def tune_for_clf_postpruning(states, labels, ccp_alphas):
    print('Searching the best ccp_alpha for clf postpruning...')

    parameters = {'ccp_alpha': ccp_alphas}
    scoring = {'accuracy', 'neg_mean_squared_error'}

    clf = tree.DecisionTreeClassifier(random_state=RANDOM_SEED)
    gs = GridSearchCV(clf, param_grid=parameters, scoring=scoring, n_jobs=-1, cv=K,
                      return_train_score=True, refit='neg_mean_squared_error')

    train_states, val_states, train_labels, val_labels = \
        train_test_split(states, labels, train_size=TRAIN_FRAC, random_state=RANDOM_SEED)
    gs.fit(train_states, train_labels)

    scores = gs.cv_results_
    index = gs.best_index_
    # print(scores.keys())
    best_perf = [scores['mean_train_accuracy'][index],
                 scores['mean_test_accuracy'][index],
                 -1 * scores['mean_train_neg_mean_squared_error'][index],
                 -1 * scores['mean_test_neg_mean_squared_error'][index]]
    bset_param = gs.best_params_
    print('Searching ends. Bset params: {}'.format(bset_param))

    return gs.best_estimator_, bset_param, best_perf


def clf_postpruning(states, labels):
    print('-' * 50)
    print('Classifier postpruning...')

    train_states, val_states, train_labels, val_labels = \
        train_test_split(states, labels, train_size=TRAIN_FRAC, random_state=RANDOM_SEED)

    if CLF_TUNE:
        clf = tree.DecisionTreeClassifier(random_state=RANDOM_SEED)
        path = clf.cost_complexity_pruning_path(train_states, train_labels)
        raw_ccp_alphas, _ = path.ccp_alphas, path.impurities
        start = lower_bound(raw_ccp_alphas, CCP_ALPHA_PARAM[0], 0)
        end = lower_bound(raw_ccp_alphas, CCP_ALPHA_PARAM[-1], 0)
        ccp_alphas = raw_ccp_alphas[start:end]
        if len(ccp_alphas) >= CPP_THRESHOLD:    # too many params
            ccp_alphas = CCP_ALPHA_PARAM
            # ccp_alphas = np.linspace(np.min(CCP_ALPHA_PARAM),
            #                          np.max(CCP_ALPHA_PARAM), CPP_THRESHOLD)
        post_clf, post_clf_param, post_clf_cv = \
            tune_for_clf_postpruning(states, labels, ccp_alphas)
    else:
        post_clf = tree.DecisionTreeClassifier(ccp_alpha=CCP_ALPHA, random_state=RANDOM_SEED)
        post_clf_param = {'ccp_alpha': CCP_ALPHA}
        post_clf.fit(train_states, train_labels)

    assert(post_clf), '[clf_postpruning] Error! There is no post_clf.'
    print('Postpruning done.')

    post_clf_t_precision = post_clf.score(train_states, train_labels)
    post_clf_v_precision = post_clf.score(val_states, val_labels)
    print('Postpruning classifier training precision: {:.3%}'.format(post_clf_t_precision))
    print('Postpruning classifier validating precision: {:.3%}'.format(post_clf_v_precision))
    post_clf_t_loss = mse(train_labels, post_clf.predict(train_states))
    post_clf_v_loss = mse(val_labels, post_clf.predict(val_states))
    print('Postpruning classifier training loss: {:.3f}'.format(post_clf_t_loss))
    print('Postpruning classifier validating loss: {:.3f}'.format(post_clf_v_loss))

    post_clf_metrics = [post_clf_t_precision, post_clf_v_precision,
                        post_clf_t_loss, post_clf_v_loss]

    # cross validation
    if not CLF_TUNE:
        scoring = {'accuracy', 'neg_mean_squared_error'}
        scores = cross_validate(post_clf, states, labels, scoring=scoring,
                                cv=K, return_train_score=True)
        # items = sorted(scores.keys())
        post_clf_cv = [np.mean(scores['train_accuracy']),
                       np.mean(scores['test_accuracy']),
                       -1 * np.mean(scores['train_neg_mean_squared_error']),
                       -1 * np.mean(scores['test_neg_mean_squared_error'])]

    print('Cross validation performance: ')
    print('CV training precision: {:.3%}'.format(post_clf_cv[0]))
    print('CV validating precision: {:.3%}'.format(post_clf_cv[1]))
    print('CV training loss: {:.3f}'.format(post_clf_cv[2]))
    print('CV validating loss: {:.3f}'.format(post_clf_cv[3]))

    return post_clf, post_clf_metrics, post_clf_param, post_clf_cv


def train_classifier(states, labels):
    # ----------unpruned clf----------
    clf, clf_metrics, clf_cv, train_cnt, val_cnt = train_unpruned_clf(states, labels)

    if CLF_PRUNE:
        # ----------prepruning (select best params)----------
        pre_clf, pre_clf_metrics, pre_clf_params, pre_clf_cv = clf_prepruning(states, labels)

        # ----------postpruning by CPP----------
        post_clf, post_clf_metrics, post_clf_param, post_clf_cv = clf_postpruning(states, labels)

        # ----------select the best classifier according to loss on validating set----------
        all_clf = [clf, pre_clf, post_clf]
        all_clf_metrics = [clf_metrics, pre_clf_metrics, post_clf_metrics]
        all_params = [None, pre_clf_params, post_clf_param]
        all_cv = [clf_cv, pre_clf_cv, post_clf_cv]
        all_loss = [metric[3] for i, metric in enumerate(all_cv)]
        min_loss = min(all_loss)

        best_index = all_loss.index(min_loss)
        best_clf = all_clf[best_index]
        best_metrics = all_clf_metrics[best_index]
        bset_params = all_params[best_index]
        best_cv = all_cv[best_index]
        print('-' * 50)
        print('Select best classifier: %s' % (TREE_NAMES[best_index]))
        return best_clf, best_metrics, best_cv, train_cnt, val_cnt, best_index, bset_params

    return clf, clf_metrics, clf_cv, train_cnt, val_cnt, None, None


def plot_feature_info(fig_dir, train_time, info, info_type):
    assert(len(info) == len(FEATURE_NAMES)), \
        '[plot_feature_info] info len error!'

    if len(info_type.split('_')) == 3:
        info_type = info_type[4:]
    str_list = [string.capitalize() for string in info_type.split('_')]
    fig_name = str_list[0] + ' ' + str_list[1]
    # plt.figure(fig_name, figsize=(3, 2), dpi=300)
    plt.figure(fig_name, dpi=300)
    titileFont = {'family': DEFAULT_FONT_FAMILY,
                  'fontweight': 'bold', 'size': 12}
    plt.title(fig_name, titileFont)

    # indices = np.argsort(info)[::-1]     # in descending order
    indices = np.argsort(info)
    feature_names = np.array(FEATURE_NAMES)

    # width = 0.3
    y_ticks = np.arange(0, len(FEATURE_NAMES)) + 0.5
    rects = plt.barh(y_ticks, info[indices], color=BAR_COLOR, align='center')
    plt.bar_label(rects, fmt='%.3f', padding=2, fontproperties=DEFAULT_FONT_FAMILY, size=8)

    y_tick_labels = feature_names[indices]
    plt.ylim(0, len(FEATURE_NAMES))
    plt.yticks(y_ticks, y_tick_labels,
               fontproperties=DEFAULT_FONT_FAMILY, size=10)
    plt.xlim(0, np.max(info) + 0.1)
    plt.xticks(fontproperties=DEFAULT_FONT_FAMILY, size=10)

    # save figure
    fig_path = fig_dir + info_type + '_' + train_time + '.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    # plt.show()

    print('Figure of {} is saved: '.format(info_type.replace('_', ' ')) + fig_path)


def export_res(train_time, **kwargs):
    tree_path = TREE_DIR + train_time + '_' + str(TARGET_TPYE) + '/'
    mkdir(tree_path)
    print('-' * 50)
    # ----------save dataset, precision and loss of tree----------
    data_info_path = tree_path + 'training_info_' + train_time + '.txt'

    with open(data_info_path, 'w+') as data_info:
        data_info.write('----------Data info----------\n')
        data_info.write('Class point: ' + str(CLASS_POINT) + '\n')
        data_info.write('Feature names: ' + str(FEATURE_NAMES) + '\n')
        if MUTUAL_INFO:
            mi = kwargs['mi']
            mis = [float('{:.4f}'.format(mic)) for mic in mi]
            data_info.write('Feature mutual information: ' + str(mis) + '\n')
        data_info.write('\n')

        if CLF_TREE:
            train_cnt = kwargs['clf_info'][0]
            val_cnt = kwargs['clf_info'][1]
            clf = kwargs['clf_info'][2]
            clf_index = kwargs['clf_info'][3]
            clf_params = kwargs['clf_info'][4]
            clf_metrics = kwargs['clf_info'][5]
            clf_cv = kwargs['clf_info'][6]

            data_info.write('----------Classifier info----------\n')
            data_info.write('Training data count: ' + str(train_cnt) + '\n')
            data_info.write('Validating data count: ' + str(val_cnt) + '\n')
            if clf_index is not None:
                data_info.write('Classifier: {}\n'.format(TREE_NAMES[clf_index]))
                if clf_params is not None:
                    param_str = 'Params: '
                    for key, value in clf_params.items():
                        param_str += '{} = {}, '.format(key, value)
                    param_str = param_str[:-2]  # remove ', '
                    param_str += '\n'
                    data_info.write(param_str)
            else:
                data_info.write('Classifier: {}\n'.format(TREE_NAMES[0]))
            clf_feature_imp = [float('{:.4f}'.format(imp)) for imp in clf.feature_importances_]
            data_info.write('Feature importances: ' + str(clf_feature_imp) + '\n')
            data_info.write('Training precision: {:.3%}\n'.format(clf_metrics[0]))
            data_info.write('Validating precision: {:.3%}\n'.format(clf_metrics[1]))
            data_info.write('Training loss: {:.3f}\n'.format(clf_metrics[2]))
            data_info.write('Validating loss: {:.3f}\n'.format(clf_metrics[3]))
            data_info.write('CV training precision: {:.3%}\n'.format(clf_cv[0]))
            data_info.write('CV validating precision: {:.3%}\n'.format(clf_cv[1]))
            data_info.write('CV training loss: {:.3f}\n'.format(clf_cv[2]))
            data_info.write('CV validating loss: {:.3f}\n\n'.format(clf_cv[3]))

        if REG_TREE:
            reg = kwargs['reg_info'][0]
            reg_index = kwargs['reg_info'][1]
            reg_params = kwargs['reg_info'][2]
            reg_metrics = kwargs['reg_info'][3]
            reg_cv = kwargs['reg_info'][4]
            data_info.write('----------Regressor info----------\n')
            if reg_index is not None:
                data_info.write('Regressor: {}\n'.format(TREE_NAMES[reg_index]))
                if reg_params is not None:
                    param_str = 'Params: '
                    for key, value in reg_params.items():
                        param_str += '{} = {}, '.format(key, value)
                    param_str = param_str[:-2]  # remove ', '
                    param_str += '\n'
                    data_info.write(param_str)
            else:
                data_info.write('Regressor: {}\n'.format(TREE_NAMES[0]))
            reg_feature_imp = [float('{:.4f}'.format(imp)) for imp in reg.feature_importances_]
            data_info.write('Feature importances: ' + str(reg_feature_imp) + '\n')
            data_info.write('Training loss: {:.3f}\n'.format(reg_metrics[0]))
            data_info.write('Validating loss: {:.3f}\n'.format(reg_metrics[1]))
            data_info.write('CV training loss: {:.3f}\n'.format(reg_cv[0]))
            data_info.write('CV validating loss: {:.3f}\n\n'.format(reg_cv[1]))

        if MLR:
            mlr_params = kwargs['mlr_info'][0]
            mlr_metrics = kwargs['mlr_info'][1]
            mlr_cv = kwargs['mlr_info'][2]
            data_info.write('----------MLR info----------\n')
            data_info.write('MLR coefficients: ' + str(mlr_params[0]) + '\n')
            data_info.write('MLR intercept: {}\n'.format(mlr_params[1]))
            data_info.write('Training coefficient of determination: {:.3f}\n'
                            .format(mlr_metrics[0]))
            data_info.write('Validating coefficient of determination: {:.3f}\n'
                            .format(mlr_metrics[1]))
            data_info.write('Training loss: {:.3f}\n'.format(mlr_metrics[2]))
            data_info.write('Validating loss: {:.3f}\n'.format(mlr_metrics[3]))
            data_info.write('CV training coefficient of determination: {:.3f}\n'.format(mlr_cv[0]))
            data_info.write(
                'CV validating coefficient of determination: {:.3f}\n'.format(mlr_cv[1]))
            data_info.write('CV training loss: {:.3f}\n'.format(mlr_cv[2]))
            data_info.write('CV validating loss: {:.3f}\n'.format(mlr_cv[3]))

    print('Dataset and training information is saved: ' + data_info_path)

    if REG_TREE:
        # ----------plot tree (tree.export_graphviz)----------
        fig_title = 'regressor_depth=' + str(PLOT_DEPTH) + '_' + train_time
        fig_path = tree_path + fig_title + '.png'

        dot_data = StringIO()
        tree.export_graphviz(reg, out_file=dot_data, max_depth=PLOT_DEPTH,
                             feature_names=FEATURE_NAMES, filled=True, rounded=True)
        out_graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        out_graph.write_png(fig_path)

        print('Figure of regressor is saved: ' + fig_path)

        # ----------export tree for python----------
        output_py_path = tree_path + 'reg_' + str(TARGET_TPYE) + '_' + train_time + '.pkl'

        # save model
        dump(reg, output_py_path)
        print('Regressor model for python is exported: ' + output_py_path)

        # export feature importances and mutual information of features
        plot_feature_info(tree_path, train_time, reg.feature_importances_,
                          'reg_feature_importances')

    if CLF_TREE:
        # ----------plot tree (tree.export_graphviz)----------
        fig_title = 'classifier_depth=' + str(PLOT_DEPTH) + '_' + train_time
        fig_path = tree_path + fig_title + '.png'

        dot_data = StringIO()
        tree.export_graphviz(clf, out_file=dot_data, max_depth=PLOT_DEPTH,
                             feature_names=FEATURE_NAMES, filled=True, rounded=True)
        out_graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        out_graph.write_png(fig_path)

        print('Figure of classifier is saved: ' + fig_path)

        # ----------export tree for js----------
        porter = Porter(clf, language='js')

        output = porter.export()
        output_js_path = tree_path + 'lumos_clf_' + train_time + '.js'
        with open(output_js_path, 'w+') as out_tree:
            out_tree.write(output)
        print('Classifier model for js is exported: ' + output_js_path)

        # ----------export tree for python----------
        output_py_path = tree_path + 'clf_' + str(TARGET_TPYE) + '_' + train_time + '.pkl'

        # save model
        dump(clf, output_py_path)
        print('Classifier model for python is exported: ' + output_py_path)

        # load model
        # clf = load(output_py_path)

        # export feature importances and mutual information of features
        plot_feature_info(tree_path, train_time, clf.feature_importances_,
                          'clf_feature_importances')

    if MUTUAL_INFO:
        plot_feature_info(tree_path, train_time, mi, 'mutual_information')


def train_models(states, labels, values, mi):
    start_time = time.localtime()
    train_time = time.strftime('%y%m%d-%H%M', start_time)

    # ----------train multiple linear regressor----------
    if MLR:
        mlr, mlr_metrics, mlr_params, mlr_cv = train_mlr(states, values)
        mlr_info = [mlr_params, mlr_metrics, mlr_cv]

    # ----------train regressor----------
    if REG_TREE:
        reg, reg_metrics, reg_cv, reg_index, reg_params = train_regressor(states, values)
        reg_info = [reg, reg_index, reg_params, reg_metrics, reg_cv]

    # ----------train classifier----------
    if CLF_TREE:
        clf, clf_metrics, clf_cv, train_cnt, val_cnt, clf_index, clf_params = \
            train_classifier(states, labels)
        clf_info = [train_cnt, val_cnt, clf, clf_index, clf_params, clf_metrics, clf_cv]

    # ----------export info, fig and model of tree----------
    if EXPORT_RES:
        kwargs = {}
        if MUTUAL_INFO:
            kwargs['mi'] = mi
        if MLR:
            kwargs['mlr_info'] = mlr_info
        if REG_TREE:
            kwargs['reg_info'] = reg_info
        if CLF_TREE:
            kwargs['clf_info'] = clf_info
        export_res(train_time, **kwargs)


if __name__ == '__main__':
    warnings.simplefilter('ignore', FutureWarning)
    from sklearn_porter import Porter

    raw_states, raw_class_labels, raw_value_labels = get_data_and_label(DATA_DIR)

    set_feature_names(raw_states)

    mi = None
    if MUTUAL_INFO:
        mi = feature_mutual_info(raw_states, raw_class_labels, raw_value_labels)

    train_models(raw_states, raw_class_labels, raw_value_labels, mi)
