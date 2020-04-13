from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def dataset_hist(data, name, path):
    validate_path(path)

    # absolute number histogram plot
    labels, counts = np.unique(data, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    plt.gca().yaxis.grid(True)
    plt.title(str(name) + ' Label Distribution')
    plt.xlabel('label name')
    plt.ylabel('number of occurrences')
    plt.savefig(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f_') + str(name) + '.png')

    # percentage histogram plot
    plt.figure(figsize=(10, 5))
    counts_percentage = [x / sum(counts) for x in counts]
    plt.bar(labels, counts_percentage, align='center')
    plt.gca().set_xticks(labels)
    plt.gca().yaxis.grid(True)
    vals = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in vals])
    plt.title(str(name) + ' Label Distribution (Percentage) - Total Labels: ' + str(sum(counts)))
    plt.xlabel('label name')
    plt.ylabel('percentage of occurrences (%)')
    plt.savefig(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f_') + str(name) + '_percentage.png')


def fit_history_plot(history, path, data_set_name=None, algo_name=None):
    validate_path(path)
    with open(path + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + 'fit_history.json', 'w') as f:
        f.write(str(history.history))
    if ('output_1_accuracy' in history.history.keys()) and ('val_output_1_accuracy' in history.history.keys()):
        plt.figure()
        plt.plot(history.history['output_1_accuracy'])
        plt.plot(history.history['val_output_1_accuracy'])
        if (data_set_name is None) and (algo_name is None):
            title = 'Label Prediction Accuracy'
        else:
            title = str(data_set_name) + ' Accuracy ' + '(' + str(algo_name) + ')'
        plt.title(title)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.gca().set_yticklabels(['{:.1%}'.format(x) for x in plt.gca().get_yticks()])
        plt.grid(linestyle='-')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.savefig(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str('label_prediction_accuracy') + '.png')
    if ('output_2_accuracy' in history.history.keys()) and ('val_output_2_accuracy' in history.history.keys()):
        plt.figure()
        plt.plot(history.history['output_2_accuracy'])
        plt.plot(history.history['val_output_2_accuracy'])
        if (data_set_name is None) and (algo_name is None):
            title = 'Reconstruction Accuracy'
        else:
            title = str(data_set_name) + ' Reconstruction Accuracy ' + '(' + str(algo_name) + ')'
        plt.title(title)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.gca().set_yticklabels(['{:.1%}'.format(x) for x in plt.gca().get_yticks()])
        plt.grid(linestyle='-')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.savefig(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str('reconstruction_accuracy') + '.png')
    if ('accuracy' in history.history.keys()) and ('val_accuracy' in history.history.keys()):
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        if (data_set_name is None) and (algo_name is None):
            title = 'Label Accuracy (without reconstruction)'
        else:
            title = str(data_set_name) + ' Accuracy ' + '(No Reconstruction ' + str(algo_name) + ')'
        plt.title(title)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.gca().set_yticklabels(['{:.1%}'.format(x) for x in plt.gca().get_yticks()])
        plt.grid(linestyle='-')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.savefig(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str('label_prediction_accuracy') + '.png')
    if 'lr' in history.history.keys():
        plt.figure()
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.ylabel('learning rate value')
        plt.xlabel('epoch')
        plt.grid(linestyle='-')
        plt.savefig(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str('learning_rate') + '.png')
    if ('output_1_loss' in history.history.keys()) and \
            ('output_2_loss' in history.history.keys()) and \
            ('loss' in history.history.keys()):
        plt.figure()
        plt.plot(history.history['output_1_loss'])
        plt.plot(history.history['output_2_loss'])
        plt.plot(history.history['loss'])
        if (data_set_name is None) and (algo_name is None):
            title = 'Loss'
        else:
            title = str(data_set_name) + ' Loss ' + '(' + str(algo_name) + ')'
        plt.title(title)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid(linestyle='-')
        plt.legend(['label loss', 'reconstruction loss', 'total loss'], loc='upper right')
        plt.savefig(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str('loss') + '.png')
    if ('loss' in history.history.keys()) and ('val_loss' in history.history.keys()):
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        if (data_set_name is None) and (algo_name is None):
            title = 'Loss'
        else:
            title = str(data_set_name) + ' Loss ' + '(' + str(algo_name) + ')'
        plt.title(title)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid(linestyle='-')
        plt.legend(['training loss', 'validation loss'], loc='upper right')
        plt.savefig(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str('loss_without_reconstruction') + '.png')


def evaluation_results(data, path):
    validate_path(path)
    d = '\n'.join(data)
    file = open(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str('evaluation_results') + '.txt', 'w')
    file.write(d)
    file.close()


def save_model_summary(model, path):
    with open("data/data_mix_300/plots/" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str('model_summary') + '.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def plot_prediction(imgs_test, labels_test_str, labels_test_pred_str, path):
    validate_path(path)

    data_dict = {}
    for t, p in zip(labels_test_str, labels_test_pred_str):
        p = p + str(' pred')
        t = t + str(' true')
        if t not in data_dict:
            data_dict[t] = {}
        if p not in data_dict[t]:
            data_dict[t][p] = 0.0
        data_dict[t][p] += 1.0

    pd.options.display.float_format = '{:.1%}'.format
    df = pd.DataFrame(data_dict)
    df = df / df.sum()
    df = df.sort_index()
    df = df.reindex(sorted(df.columns), axis=1)
    df.to_pickle(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + 'prediction.pkl')

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title('Percentage of correct predictions')
    chart = sns.heatmap(df, cmap='Blues', annot=True, vmin=0., vmax=1., cbar=True, ax=ax)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=30, horizontalalignment='right')

    plt.savefig(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str('prediction') + '.png')


def validate_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def store_config(name, config, path):
    validate_path(path)
    data = str(name) + '\n' + str(config)
    with open(str(path) + datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str('config') + '.txt', 'w') as f:
        f.write(data)
