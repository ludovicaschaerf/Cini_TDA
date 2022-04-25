import pickle
from model_replica import ReplicaNet
import numpy as np
from utils import preprocess_image_test, catch
from tqdm import tqdm
from scipy import sparse

replica_dir = '/mnt/project_replica/datasets/cini/'

data_dir = '/scratch/students/schaerf/'

with open(data_dir + 'list_iconography.pkl', 'rb') as infile:
    files = pickle.load(infile)

print(len(files))

with open(data_dir + 'uid2path.pkl', 'rb') as outfile:
        uid2path = pickle.load(outfile)


model = ReplicaNet('resnext-101', 'cpu')
    
embeddings = []
for img in tqdm(files):
    C = preprocess_image_test(replica_dir + catch(img, uid2path), 320)
    pool_C = model.predict_non_pooled(C)
    pool_C = np.moveaxis(pool_C.squeeze(0).cpu().detach().numpy(), 0, -1)

    embeddings.append([img, pool_C]) #sparse.csr_matrix(
    

embeddings = np.array(embeddings, dtype=np.ndarray)
#print(embeddings)

np.save(data_dir + 'embedding_no_pool/' 'madonnas.npy', embeddings)