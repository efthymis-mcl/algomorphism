import numpy as np
import matplotlib.pyplot as plt


def pca_denoising_figure(pca_vl_s, pca_ts_s, pca_vl_u, pca_ts_u, pca_emb, knn_pca,
                         zlabels, pca_emb_idxs, save_obj=None):
    """
    Draw the knn_pca spaces, plot target embeddings, predicted validation (seen) embeddings and test (unseen) embeddings.

    Args:
        pca_vl_s: A ndarray, pca denoised validation seen embeddings where predicted,
        pca_ts_s: A ndarray, pca denoised test seen embeddings where predicted,
        pca_vl_u: A ndarray, pca denoised validation unseen unseen embeddings where predicted,
        pca_ts_u: A ndarray, pca denoised test unssen embeddings where predicted,
        pca_emb: A ndarray, pca denoised true embeddings,
        knn_pca: A KNeighborsClassifier object, trained by pca_emb,
        zlabels: A list, list of true embedding labels,
        pca_emb_idxs: A list (Optional), list of subpaths saving figure. Default is [0,1,2],
        save_obj: A list (Optional), list of subpaths saving figure. Default is None.

    """
    dpi = 100
    xmin = np.min(pca_emb[:, 0])
    xmin += np.sign(xmin)
    xmax = np.max(pca_emb[:, 0])
    xmax += np.sign(xmax)
    ymin = np.min(pca_emb[:, 1])
    ymin += np.sign(ymin)
    ymax = np.max(pca_emb[:, 1])
    ymax += np.sign(ymax)

    xlin = np.linspace(xmin, xmax, dpi)
    ylin = np.linspace(ymin, ymax, dpi)
    xx, yy = np.meshgrid(xlin, ylin)
    knn_space = np.argmax(knn_pca.predict(np.c_[xx.ravel(), yy.ravel()]), axis=1)
    knn_space = knn_space.reshape(xx.shape)

    Dx = 0
    Dy = 1

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(pca_emb[pca_emb_idxs, Dx], pca_emb[pca_emb_idxs, Dy], 'o', label='embs', markersize=15)
    plt.plot(pca_vl_s[:, Dx], pca_vl_s[:, Dy], '*', label='val seen', markersize=10)
    plt.plot(pca_vl_u[:, Dx], pca_vl_u[:, Dy], '+', label='val unseen', markersize=10)
    plt.plot(pca_ts_s[:, Dx], pca_ts_s[:, Dy], 'v', label='test seen', markersize=10)
    plt.plot(pca_ts_u[:, Dx], pca_ts_u[:, Dy], 's', label='test unseen', markersize=10)

    zlabels = zip(zlabels, range(len(zlabels)))
    zlabels = list(filter(lambda x: x[0] if x[1] in pca_emb_idxs else None, zlabels))
    zlabels = [zlabel[0] for zlabel in zlabels]

    for (v, l) in zip(pca_emb[pca_emb_idxs], zlabels):
        plt.text(v[Dx], v[Dy], l, fontsize=20)

    ax.contourf(xx, yy, knn_space, cmap=plt.get_cmap('tab20c'), levels=len(pca_emb_idxs))
    plt.legend()
    if save_obj is not None:
        plt.savefig('{}/{}.eps'.format(*save_obj), format='eps')
    else:
        plt.show()


def multiple_models_history_figure(nn_models: list, figsize=(16, 8), legend_fontsize=18, axes_label_fondsize=22,
                                   ticks_fontsize=16, save_obj=None):
    """
    Args:
        nn_models: A list, a model with interference ` algomorphism.base.BaseNueralNetwork `  object,
        figsize: A tuple or list, figure size of x and y axis,
        legend_fontsize: A int,
        axes_label_fondsize: A int,
        ticks_fontsize: A int,
        save_obj: A list, list of path and name of saving figure.

    """
    def score_cost_plot(history:dict, nn_model_name:str, plot_type='cost'):
        for key, v in history.items():
            key_split = key.split('_')
            if any(v) and plot_type in key_split:
                epoch_v = [k for k in v.keys()]
                score_cost_v = [value for value in v.values()]
                if plot_type != 'harmonic':
                    key_split.remove(plot_type)
                label = ' \, '.join(key_split)
                plt.plot(epoch_v, score_cost_v, label=r'${}: \,{}$'.format(nn_model_name, label))

        plt.xlabel(r"$\# \, of \, epochs$", fontsize=axes_label_fondsize)
        plt.ylabel(r'${}$'.format(plot_type), fontsize=axes_label_fondsize)
        plt.legend(fontsize=legend_fontsize)
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)

    plt.figure(figsize=figsize)
    for nn_model in nn_models:
        plt.subplot(1, 2, 1)
        score_cost_plot(nn_model.history, nn_model.name, 'cost')
        plt.subplot(1, 2, 2)
        score_cost_plot(nn_model.history, nn_model.name, 'score')
        score_cost_plot(nn_model.history, nn_model.name, 'harmonic')

    if save_obj is not None:
        plt.savefig('{}/{}.eps'.format(*save_obj), format='eps')
    else:
        plt.show()


# def print_confmtx(model, dataset, lerninfo: str, indexes_list: list):
#     """
#     Print confusion matrix per examples (train, test, validation)
#     Args:
#         model: An object, The model interference ` algomorphism.base.BaseNueralNetwork ` object,
#         dataset: An object,
#         lerninfo: A str,
#         indexes_list: A list.
#     """
#
#     def confmtx(y, yhat):
#         confmtx = np.zeros((y.shape[1], y.shape[1]))
#         y = np.argmax(y, axis=1)
#         yhat = np.argmax(yhat, axis=1)
#         y_concat = np.concatenate([y.reshape(-1, 1), yhat.reshape(-1, 1)], axis=1)
#
#         for y1, y2 in y_concat:
#             confmtx[y1, y2] += 1
#
#         return confmtx
#
#     (_, outs_tr, p_train), (_, outs_vl, p_val), (_, outs_ts, p_test) = model.get_results(dataset, True)
#
#     indxr = indexes_list[0]
#     indxa = indexes_list[1]
#
#     (r_train, a_train) = outs_tr[indxr], outs_tr[indxa]
#     (r_val, a_val) = outs_vl[indxr], outs_vl[indxa]
#     (r_test, a_test) = outs_ts[indxr], outs_ts[indxa]
#
#     lr = [[r_train, p_train[indxr]], [r_val, p_val[indxr]], [r_test, p_test[indxr]]]
#     la = [[a_train, p_train[indxa]], [a_val, p_val[indxa]], [a_test, p_test[indxa]]]
#
#     print('a', lerninfo)
#     for expl, (tr, pr) in zip(['train', 'val', 'test'], la):
#         print(expl)
#         print(confmtx(tr, pr))
#
#     print('r', lerninfo)
#     for expl, (tr, pr) in zip(['train', 'val', 'test'], lr):
#         print(expl)
#         print(confmtx(tr, pr))
