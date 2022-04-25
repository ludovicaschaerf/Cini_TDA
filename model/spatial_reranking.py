from model_replica import ReplicaNet
import numpy as np
from time import time
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import multiprocessing as mp
    
def preprocess_image_test(img_name, resolution=480):
    img = Image.open(img_name)
    tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.RandomResizedCrop((resolution, resolution), ),
            transforms.Resize((resolution, resolution)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tfms(img).unsqueeze(0)


def catch(x, uid2path):
    try:
        return uid2path[x]
    except:
        return np.nan


def rerank_spatial(uid, sims, uid2path):
    model = ReplicaNet('resnext-101', 'cpu')
    replica_dir = '/mnt/project_replica/datasets/cini/'
    A = preprocess_image_test(replica_dir + catch(uid, uid2path), 320)
    pool_A = model.predict_non_pooled(A)
    pool_A = np.moveaxis(pool_A.squeeze(0).cpu().detach().numpy(), 0, -1)
    
    print(pool_A.shape)
    Cs = [preprocess_image_test(replica_dir + catch(uid, uid2path), 320) for uid in tqdm(sims)]
    pool_Cs = [model.predict_non_pooled(C) for C in tqdm(Cs)]
    
    print(pool_Cs[0].shape)
    
    #pool = mp.Pool(mp.cpu_count() - 20)
    #ranks = pool.starmap(match_feature_maps_simple, [(pool_A, np.moveaxis(pool_c.squeeze(0).cpu().detach().numpy(), 0, -1)) for pool_c in pool_Cs])
    #pool.close()
    
    ranks = [match_feature_maps_simple(pool_A, np.moveaxis(pool_c.squeeze(0).cpu().detach().numpy(), 0, -1)) for pool_c in tqdm(pool_Cs)]
    sort_arr = np.argsort(ranks)
    rev_arr = np.flipud(sort_arr) 
    sims_rerank = np.array(sims)[rev_arr]
    return sims_rerank

class Timer:

    def __init__(self, description, disable=False):
        self.description = description
        self.disable = disable

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.interval = self.end - self.start
        if not self.disable:
            print("{} : {}".format(self.description, self.interval))


def normalize(f_map, norm_epsilon):
    for i in range(f_map.shape[0]):
        for j in range(f_map.shape[1]):
            f_map[i, j, :] /= np.linalg.norm(f_map[i, j]) + norm_epsilon


def nb_unravel(ind, dims):
    result = np.empty(len(dims), dtype=np.int32)
    offsets = np.empty(len(dims), dtype=np.int32)
    d = len(dims)
    o = 1
    for i in range(d):
        offsets[d-1-i] = o
        o *= dims[d-1-i]
    for i, o in enumerate(offsets):
        result[i] = ind // o
        ind = ind % o
    return result

def nb_unravel_array(inds, dims): # not sure what this tries to do
    result = np.empty((len(inds), len(dims)), dtype=np.int32)
    offsets = np.empty(len(dims), dtype=np.int32)
    d = len(dims)
    o = 1
    for i in range(d):
        offsets[d-1-i] = o
        o *= dims[d-1-i]
    for k in range(len(inds)):
        ind = inds[k]
        for i, o in enumerate(offsets):
            result[k, i] = ind // o
            ind = ind % o
    return result


def get_candidates(f_map_1, f_map_2, norm_epsilon=0, margin=1, crosscheck_limit=2):
    normalize(f_map_1, norm_epsilon) # can easily be added into a model
    normalize(f_map_2, norm_epsilon)

    h1, w1, d_size = f_map_1.shape

    h2, w2, _ = f_map_2.shape
    # Convert to descriptors, keypoint versions
    des1 = np.ascontiguousarray(f_map_1[margin:h1 - margin, margin:w1 - margin]).reshape((-1, d_size)) #remove a margin: from 10x10 -> 8x8
    des2 = np.ascontiguousarray(f_map_2[margin:h2 - margin, margin:w2 - margin]).reshape((-1, d_size))
    # print(des1.shape) 64x2048
    kp1 = (nb_unravel_array(np.arange(len(des1)), (h1 - 2 * margin, w1 - 2 * margin)) + 0.5 + margin).astype(np.float32) # why +1.5?
    kp2 = (nb_unravel_array(np.arange(len(des2)), (h2 - 2 * margin, w2 - 2 * margin)) + 0.5 + margin).astype(np.float32)
    # print(kp1.shape) 64x2
    # print(kp1[0]) 1.5 1.5
    # Because of the margin, the arrays might be empty
    if len(kp1) == 0 or len(kp2) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32),  np.empty((0,), np.float32)
    # d = distance.cdist(des1, des2)
    d = 1 - des1 @ des2.T  # Warning, that is ~1/2 of the euclidean distance since des1.norm ~ des2.norm ~ 1
    # computes inner product distance between all 64 2048 dimensional descriptors and all 64 others
    # print(d.shape) 64x64
    #print(d[0])
    best_1 = np.empty((len(d), crosscheck_limit), dtype=np.int32)
    for i in range(len(d)):
        best_1[i, :] = np.argsort(d[i, :])[:crosscheck_limit] # for each row get two most similar ones
    d_T = d.T
    best_2 = np.empty((len(d_T), crosscheck_limit), dtype=np.int32)
    for i in range(len(d_T)):
        best_2[i, :] = np.argsort(d_T[i, :])[:crosscheck_limit] # for each column get two most similar 

    # best_1 = np.argsort(d, axis=1)[:, :crosscheck_limit]
    # best_2 = np.argsort(d.T, axis=1)[:, :crosscheck_limit]
    best_1_o = [set(best_1[i, :]) for i in range(len(best_1))]
    best_2_o = [set(best_2[i, :]) for i in range(len(best_2))]
    # d = des1 @ des2.T
    # best_1 = np.argmax(d, axis=1)
    # best_2 = np.argmax(d, axis=0)
    good = np.array([(i, j) for i, s in enumerate(best_1_o) for j in s if i in best_2_o[j]])
    
    src_pts = kp1[good[:, 0]]
    dst_pts = kp2[good[:, 1]]
    distances = np.array([d[good[i, 0], good[i, 1]] for i in range(len(good))])
    return src_pts, dst_pts, distances


def spatially_coherent_mask(src_pts, dst_pts, residual_threshold=2.0):
    min_x0_y0, increment_x0_y0 = -15, 1 # gets possible modifications
    max_x0_y0 = -min_x0_y0
    possible_x0 = np.arange(min_x0_y0, max_x0_y0, increment_x0_y0, np.int32) # -15, -14, ... , 15
    possible_y0 = np.arange(min_x0_y0, max_x0_y0, increment_x0_y0, np.int32)
    possible_lambdas = np.exp(np.arange(-7, 7)*0.2)  # ln(4) ~ 1.4 so 0.25-4x zoom
    possible_params = np.zeros((len(possible_x0), len(possible_y0), len(possible_lambdas), 2)) # Dx, Dy, lambda, flip
    for i in range(len(src_pts)):
        src_y, src_x = src_pts[i]
        dst_y, dst_x = dst_pts[i]
        for i_lamb, lamb in enumerate(possible_lambdas): # lambda is shift?
            x0 = dst_x-lamb*src_x
            y0 = dst_y-lamb*src_y
            i_x0 = int(round((x0-min_x0_y0)/increment_x0_y0))
            i_y0 = int(round((y0-min_x0_y0)/increment_x0_y0))
            if 0 <= i_x0 < len(possible_x0) and 0 <= i_y0 < len(possible_y0):
                possible_params[i_x0, i_y0, i_lamb, 0] += 1

            # Flip
            x0 = dst_x+lamb*src_x
            i_x0 = int(round((x0-min_x0_y0)/increment_x0_y0))
            if 0 <= i_x0 < len(possible_x0) and 0 <= i_y0 < len(possible_y0):
                possible_params[i_x0, i_y0, i_lamb, 1] += 1

    best_inds = np.argsort(possible_params.ravel())[-5:]
    best_inliers = 0
    best_mask = np.full(len(src_pts), False, dtype=np.bool_)
    best_M = np.array([
                [1, 0],
                [0, 1],
                [0, 0]
            ], dtype=np.float32)
    mask = best_mask.copy()
    preds = np.empty((1, 2), dtype=np.float32)
    src_pts_intercept = np.concatenate((src_pts, np.ones((len(src_pts), 1), dtype=np.float32)), axis=1)
    for ind in best_inds:
        i_x0, i_y0, i_lamb, is_flipped = nb_unravel(ind, dims=possible_params.shape)
        x0 = possible_x0[i_x0]
        y0 = possible_y0[i_y0]
        lamb = possible_lambdas[i_lamb]

        M = np.array([
                [lamb, 0],
                [0, (1-2*int(is_flipped))*lamb],
                [y0, x0]
            ], dtype=np.float32)
        preds = src_pts_intercept @ M
        mask = np.sum(np.square(preds-dst_pts), axis=1) <= residual_threshold
        if np.sum(mask) > best_inliers:
            best_inliers = np.sum(mask)
            best_mask[:] = mask
            best_M = M

    assert np.sum(best_mask) == best_inliers, "weird"

    if best_inliers > 0:
        assert np.sum(best_mask) > 0, "weird2"
        # Refine matrix
        #M, _, _, _ = np.linalg.lstsq(np.concatenate([src_pts[best_mask], np.ones((np.sum(best_mask), 1))], axis=1),
        #                             dst_pts[best_mask], rcond=-1)
        return best_M, best_mask
    else:
        print("No inliers?")
        return best_M, best_mask


def match_feature_maps(f_map_1, f_map_2, norm_epsilon=0, margin=1, crosscheck_limit=3):
    with Timer("candidates", disable=True):
        src_pts, dst_pts, distances = get_candidates(f_map_1, f_map_2, norm_epsilon, margin, crosscheck_limit)
    
    if len(src_pts) == 0:
        print("No candidates")

    with Timer("spatially_coherent", disable=True):
        M, mask = spatially_coherent_mask(src_pts, dst_pts, residual_threshold=2.0)
    num_matches = int(np.sum(mask))

    h1, w1, _ = f_map_1.shape
    h2, w2, _ = f_map_2.shape

    if num_matches == 0:
        return num_matches, None, (src_pts, dst_pts), mask.tolist(), ((0,0,1,1), (0,0,1,1))

    m1 = np.min(src_pts[mask], axis=0)
    m2 = np.max(src_pts[mask], axis=0)
    box1 = ((m1[0]-0.5)/h1, (m1[1]-0.5)/w1, (m2[0]-m1[0]+1)/h1, (m2[1]-m1[1]+1)/w1)
    m1 = np.min(dst_pts[mask], axis=0)
    m2 = np.max(dst_pts[mask], axis=0)
    box2 = ((m1[0]-0.5)/h2, (m1[1]-0.5)/w2, (m2[0]-m1[0]+1)/h2, (m2[1]-m1[1]+1)/w2)

    return num_matches, None, (src_pts, dst_pts), mask.tolist(), (box1, box2)


def match_feature_maps_simple(f_map_1, f_map_2, norm_epsilon=0, margin=1, crosscheck_limit=3):
    with Timer("candidates", disable=True):
        src_pts, dst_pts, distances = get_candidates(f_map_1, f_map_2, norm_epsilon, margin, crosscheck_limit)
    
    if len(src_pts) == 0:
        print("No candidates")

    with Timer("spatially_coherent", disable=True):
        M, mask = spatially_coherent_mask(src_pts, dst_pts, residual_threshold=2.0)
    num_matches = int(np.sum(mask))
    return num_matches 
