import numpy as np
from scipy.spatial.distance import cosine, cdist
def getPlace(query_label, sorted_gallery_labels):    
    place = 0
    for i in xrange(len(sorted_gallery_labels)):
        if  query_label == sorted_gallery_labels[i]:
            return place
        else :
            place += 1
    return place 


def ranking(descrs_query, query_labels, descrs_gallery, gallery_labels, maxrank=50):
    ranks = np.zeros(maxrank+1)
    places = dict()
    all_dist = cdist(np.array(descrs_query), np.array(descrs_gallery), metric='cosine')
    np_gallery_labels = np.array(gallery_labels)
    all_gallery_labels_sorted = np_gallery_labels[np.argsort(all_dist).astype(np.uint32)]
    for qind in xrange(len(descrs_query)):       
        dist = all_dist[qind]
        gallery_labels_sorted  = all_gallery_labels_sorted[qind]
        place=getPlace(query_labels[qind], gallery_labels_sorted)       
        places[qind] = place
        ranks[place+1:maxrank+1] += 1
    return ranks[1:]*1.0/len(descrs_query)

