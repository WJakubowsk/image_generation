import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm

from inception import InceptionV3

DIM = 2048
device = torch.device("cuda:0")


def torch_cov(m, rowvar=False):
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def sqrt_newton_schulz(A, numIters, dtype=None):
    with torch.no_grad():
        if dtype is None:
            dtype = A.type()
        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
        K = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1)
        Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1)
        K = K.type(dtype)
        Z = Z.type(dtype)
        for i in range(numIters):
            T = 0.5 * (3.0 * K - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6, use_torch=False):
    if use_torch:
        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"
        diff = mu1 - mu2
        covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50)
        if torch.any(torch.isnan(covmean)):
            return float("nan")
        covmean = covmean.squeeze()
        out = (
            (
                diff.dot(diff)
                + torch.trace(sigma1)
                + torch.trace(sigma2)
                - 2 * torch.trace(covmean)
            )
            .cpu()
            .item()
        )
    else:
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        out = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return out


def get_statistics(
    images,
    num_images=None,
    batch_size=50,
    use_torch=False,
    verbose=False,
    parallel=False,
):
    if num_images is None:
        try:
            num_images = len(images)
        except:
            raise ValueError(
                "when `images` is not a list like object (e.g. generator), "
                "`num_images` should be given"
            )

    block_idx1 = InceptionV3.BLOCK_INDEX_BY_DIM[DIM]
    model = InceptionV3([block_idx1]).to(device)
    model.eval()

    if parallel:
        model = torch.nn.DataParallel(model)

    if use_torch:
        fid_acts = torch.empty((num_images, DIM)).to(device)
    else:
        fid_acts = np.empty((num_images, DIM))

        iterator = iter(
            tqdm(
                images,
                total=num_images,
                dynamic_ncols=True,
                leave=False,
                disable=not verbose,
                desc="get_inception_and_fid_score",
            )
        )

        start = 0
        while True:
            batch_images = []
            try:
                for _ in range(batch_size):
                    batch_images.append(next(iterator))
            except StopIteration:
                if len(batch_images) == 0:
                    break
                pass
            batch_images = np.stack(batch_images, axis=0)
            end = start + len(batch_images)
            batch_images = images
            batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
            batch_images = batch_images.to(device)
            with torch.no_grad():
                pred = model(batch_images)
                if use_torch:
                    fid_acts[start:end] = pred[0].view(-1, DIM)
                else:
                    fid_acts[start:end] = pred[0].view(-1, DIM).cpu().numpy()
            start = end

    if use_torch:
        m1 = torch.mean(fid_acts, axis=0)
        s1 = torch_cov(fid_acts, rowvar=False)
    else:
        m1 = np.mean(fid_acts, axis=0)
        s1 = np.cov(fid_acts, rowvar=False)
    return m1, s1


def get_fid_score(
    _,
    images,
    num_images=None,
    batch_size=50,
    use_torch=False,
    verbose=False,
    parallel=False,
    reference_images=None,
):
    m1, s1 = get_statistics(
        images, num_images, batch_size, use_torch, verbose, parallel
    )
    m2, s2 = get_statistics(
        reference_images, batch_size=batch_size, use_torch=use_torch, verbose=verbose
    )
    if use_torch:
        m2 = torch.tensor(m2).to(m1.dtype)
        s2 = torch.tensor(s2).to(s1.dtype)
    fid_value = (
        calculate_frechet_distance(m1, s1, m2, s2, use_torch=use_torch).cpu().item()
    )
    return fid_value
