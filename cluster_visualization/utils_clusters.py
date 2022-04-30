import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image


def draw_clusters(cluster_num, cluster_df, data):
    rows = cluster_df[cluster_df['cluster'] == cluster_num]
    n = rows.shape[0]
    
    f, axarr = plt.subplots((n // 4 + 1),4, figsize=(30,7* n // 4 + 1))
    
    plt.suptitle('Cluster n '+ str(cluster_num))
    axarr = axarr.flatten()
    for i,row in enumerate(rows.iterrows()):
        row_2 = data[data['uid'] == row[1]['uid']]
        info_2 = str(row_2["AuthorOriginal"].values[0]) + '\n ' + str(row_2["Description"].values[0])
    
        axarr[i].set_title(info_2)
        drawer = row[1]['path'].split('/')[0]
        img = row[1]['path'].split('_')[1].split('.')[0]
        image = requests.get(f'https://dhlabsrv4.epfl.ch/iiif_replica/cini%2F{drawer}%2F{drawer}_{img}.jpg/full/300,/0/default.jpg')
        
        axarr[i].imshow(Image.open(BytesIO(image.content))) #replica_dir + 
        
    plt.savefig('../figures/cluster_' + str(cluster_num) + '.jpg')
    plt.show()
    