from typing import Union, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import Tensor
from algomorphism.base import BaseNeuralNetwork


def sift_point_to_best(target_point: Union[np.ndarray, Tensor], point: Union[np.ndarray, Tensor], sift_dist: float) -> Tuple[Union[np.ndarray, Tensor], Union[np.ndarray, Tensor]]:
    """
    Move a point to target point given a distance. Based on Jensen's inequality formula.

    Args:
        target_point (`Union[np.ndarray, Tensor]`): the target point of pca,
        point (`Union[np.ndarray, Tensor]`): point of pca,
        sift_dist (`float`): distance where point will sift to new one.

    Returns:
        `Tuple[Union[np.ndarray, Tensor], Union[np.ndarray, Tensor]]`: new_points, a couple of new updated points.

    References:
        https://en.wikipedia.org/wiki/Jensen%27s_inequality
    """

    dist = np.sqrt(np.sum((point - target_point) ** 2))
    a = sift_dist / dist
    new_point = np.array([
        point[0] * a + (1 - a) * target_point[0],
        point[1] * a + (1 - a) * target_point[1]
    ])
    new_points = (new_point[0], new_point[1])
    return new_points


def pca_denoising(p_examples: Union[np.ndarray, Tensor], pca_emb_examples: Union[np.ndarray, Tensor],
                  pca_emb_pred: Union[np.ndarray, Tensor], knn: KNeighborsClassifier, knn_pca: KNeighborsClassifier,
                  dx: int = 0, dy: int = 1):
    """
    PCA de-nosing method.

    Args:
        p_examples (`Union[np.ndarray, Tensor]`): high dimensional data embeddings of target points,
        pca_emb_examples (`Union[np.ndarray, Tensor]`): low dimensional pca data embeddings of target points,
        pca_emb_pred (`Union[np.ndarray, Tensor]`): low dimensional pca data embeddings of prediction.
        knn (`KNeighborsClassifier`): knn model trained by p_examples,
        knn_pca (`KNeighborsClassifier`): knn model trained by pca_emb_examples,
        dx (`int`): index dimension of x-axis,
        dy (`int`): index dimension of y-axis.

    Returns:
        pca_emb_pred: A ndarray or tensor, low dimensional updated pca data embeddings of prediction.

    See Also:
        After pca process the real distance between pca examples and pca embeddings have noise by dimension reduction.
    """
    diferent_cls_ts = np.where(knn_pca.kneighbors(pca_emb_pred[:, [dx, dy]])[1] != knn.kneighbors(p_examples)[1])[0]
    knn_n = knn.kneighbors(p_examples)[1]
    for ii in diferent_cls_ts:
        sift_dist = knn_pca.kneighbors(pca_emb_pred[:, [dx, dy]])[0][ii][0]

        x, y = sift_point_to_best(pca_emb_examples[knn_n[ii][0], [dx, dy]], pca_emb_pred[ii, [dx, dy]], sift_dist)
        pca_emb_pred[ii, dx] = x
        pca_emb_pred[ii, dy] = y

    return pca_emb_pred


def pca_denoising_preprocessing(model: BaseNeuralNetwork, dataset: object, z: Tensor, y: Tensor, emb_idx: int = 0,
                                pca_emb_idxs: list = None, example_predicted_types: list = None) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, KNeighborsClassifier]:
    """
    Args:
        model (`BaseNeuralNetwork`): Model Object (MO),
        dataset (`object`): Zero Shot Dataset object
        z (`Tensor`): true embeddings,
        y (`Tensor`): predicted embeddings,
        emb_idx (`int`): embedding index, default is 0,
        pca_emb_idxs (`list`): PCA embedding indexes, default is [0,1,2],
        example_predicted_types (`list`): example predicted types which be plotted, default is ['train'].

    Returns:
        `Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, KNeighborsClassifier]` :
            (pca_vl_s),  pca de-noised validation seen embeddings where predicted,
            (pca_ts_s): pca de-noised test seen embeddings where predicted,
            (pca_vl_u): pca de-noised validation unseen, unseen embeddings where predicted,
            (pca_ts_u): pca de-noised test unseen embeddings where predicted,
            (pca_emb): PCA de-noised true embeddings,
            (knn_pca), PCA KNeighborsClassifier object.
    """
    if example_predicted_types is None:
        example_predicted_types = ['train']
    results_dict = model.get_results(dataset,  is_score=False)

    pca = PCA(n_components=2)
    pca.fit(z)

    pca_emb = pca.transform(z)
    pca_predicted_types = {}
    for ept in example_predicted_types:
        pca_ept = pca.transform(results_dict[ept]["predicts"][emb_idx])
        pca_predicted_types[ept] = pca_ept

    dx = 0
    dy = 1
    knn_pca = KNeighborsClassifier(1)
    knn_pca.fit(pca_emb[:, [dx, dy]][pca_emb_idxs], y[pca_emb_idxs])

    for k in pca_predicted_types.keys():
        pca_dn_ept = pca_denoising(results_dict[k]["predicts"][emb_idx], pca_emb, pca_predicted_types[k], model.knn, knn_pca)
        pca_predicted_types[k] = pca_dn_ept

    return pca_predicted_types, pca_emb, knn_pca


def three_d_identity_matrix(n: int) -> np.ndarray:
    """
    Return a 3D identity matrix.

    Args:
        n (`int`): size of 3d matrix.

    Returns:
        `np.ndarray`: 3D identity matrix.
    """
    return np.array([[[1 if i == j and j == w else 0 for i in range(n)] for j in range(n)] for w in range(n)])
