# Semi-supervised Clustering of Visual Signatures of Artworks
## A human-in-the-loop approach to tracing visual pattern propagation in art history using deep computer vision methods.
### Main repository

**Basic information**
- Student name: Ludovica Schaerf
- Supervisors: Paul Guhennec (PhD), Frederic Kaplan (Prof.)
- Year: 2021-2022

**About**: The project Replica, about six years ago, paved the way to computational studies of visual patterns in art history. Simultaneously, it created the possibility for art historians to trace the propagation of patterns throughout the history of art. During the project, an image retrieval network was set up to discover artistic patterns given an input image. Despite successfully serving monographic needs and targeted search attempts, the network does not propose spontaneous discoveries. In this thesis, we eliminate the middle man of the input image, creating clusters of artworks sharing a common pattern propagation. The clusters are integrated further with a 2D coordinate-based visualisation, which provides an organic view of the evolution of the patterns in art history. 

In this effort, we demonstrate the effectiveness of fine-tuning deep learning models on a set of visual connections using a compound Hinge loss and ResNeXt architecture. Moreover, we show that clustering the trained visual signatures with OPTICS yields remarkable precision. We emphasise the importance of the semi-supervised learning of the clusters, proving the qualitative and quantitative improvement over generic clustering methods. Furthermore, we close the loop of the semi-supervised clustering through the annotation of the new findings in the clusters proposed, and retraining thereof. In total, we add over 700 new images to the set of slightly over 1800 existing visual connections. We find, in addition, examples of cross-domain, architectural, design and sketch based patterns, which were previously outside the scope of the known visual connections.


**Research summary**: 
The base architecture

Among the top competitors of ILSVRC, this paper uses experiments with different architectures and their pre-trained weights as starting points for the retrieval model: ResNet, ResNeXt, and EfficientNet (\cite{resnet, xie_aggregated_2017, tan_efficientnet_2020}). The saved architectures and their weights are downloaded and imported using the \texttt{torchvision} module and the models are then fine-tuned on the morphograph.
Based on the results by \cite{seguin_visual_2016, babenko2014neural}, we use the architectures mentioned above until their last convolutional layer. We include a mean global pooling layer as:
$$f_{mean}(I)[l] = \sum_{j,k} F_{j,k,l}$$ 
and normalise the descriptor with L2 normalisation as:
$$ f_I = f_{norm}(I) = \frac{f_{mean}(I)}{||f_{mean}(I)||^2}$$ 
The normalisation creates a descriptor that is suited for the similarity computation of image retrieval. 
The loss adopted in this thesis incorporates to the standard loss the anchor swap technique and intra-sample penalisation (\cite{balntas_learning_2016, ho_learning_2021}). Specifically, the \textit{anchor swap} defines the negative distance as $d^- = min(d(A_i, C_i), d(B_i, C_i))$. It considers the distance to the negative sample as the lowest distance between $A_i, C_i$ and $B_i, C_i$, thus always taking in consideration the negative distance that yields the greatest learning potential. On the other side, the \textit{intra-sample penalisation} adds a second margin loss to the training. This minimises the positive distance to be below a margin $m_2$ as $\max(0, m_2 − d^+)$. 

Including the two additions to the Hinge loss, final loss becomes:

$$\mathcal{L} =  \sum_{i}{max(0, m + d^+ - d^- + \max(0, d^+ - m_2))} = $$
$$ = \sum_{i}{max(0, m + d(A_i, B_i) - \min(d(A_i, C_i), d(B_i, C_i)) + \max(0, d(A_i, B_i) - m_2))}$$

include a brief summary of your approaches/implementations and an illlustration of your results.

**Installation and Usage**
- dependencies: platform, libraries (for Python include a `requirements.txt` file)
- compilation (if necessary)
- usage: how to run your code

**License**    
    We encourage you to choose an open license (e.g. AGPL, GPL, LGPL or MIT).    
    License files are already available in GH (add new file, start typing license, choices will appear).    
    You can also add the following at the end of your README:       
   
	    semi-supervised-clustering-of-visual-signatures-of-artworks - Ludovica Schaerf    
	    Copyright (c) 2022 EPFL    
	    This program is licensed under the terms of the [license]. 
	    
