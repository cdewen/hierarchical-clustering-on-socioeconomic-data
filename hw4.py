import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hierarchy

def load_data(filepath):
    output = []
    
    csv.DictReader(open(filepath, 'r'))
    for row in csv.DictReader(open(filepath, 'r')):
        output.append(row)

    return output

def calc_features(row):
    values = list(row.values())
    values = values[2:]
    ar = np.array(values, dtype=np.float64)
    return ar

def hac(features):
    distance = np.zeros((len(features), len(features)))

    for i in range(len(features)):
        for j in range(len(features)):
            if i == j:
                distance[i][j] = np.inf
            else:
                distance[i][j] = np.linalg.norm(features[i]-features[j])

    clusters = {i: [i] for i in range(len(features))}
    cluster_index = {i: i for i in range(len(features))}

    n = len(features)
    Z = np.zeros((n-1, 4))

    for i in range(n-1):
        min_index = np.where(distance == np.min(distance))[0]

        Z[i][0] = min(cluster_index[min_index[0]],cluster_index[min_index[1]])
        Z[i][1] = max(cluster_index[min_index[0]],cluster_index[min_index[1]])
        Z[i][2] = distance[min_index[0]][min_index[1]]
        Z[i][3] = len(clusters[min_index[0]]) + len(clusters[min_index[1]])

        newIndex = min(min_index[0], min_index[1])
        replaceIndex = max(min_index[0], min_index[1])

        cluster_index[newIndex] = i + n
        clusters[newIndex] = clusters[min_index[0]] + clusters[min_index[1]]

        for j in range(n):
            distance[newIndex][j] = max(distance[min_index[0]][j], distance[min_index[1]][j])
            distance[j][newIndex] = max(distance[j][min_index[0]], distance[j][min_index[1]])
            distance[replaceIndex][j] = np.inf
            distance[j][replaceIndex] = np.inf
        
    return Z 

def fig_hac(Z, name):
    fig = plt.figure()
    dn = hierarchy.dendrogram(Z, labels=name, leaf_rotation=90)
    fig.tight_layout    
    return fig

def normalize_features(features):
    columnMean = np.mean(features, axis=0)
    columnStd = np.std(features, axis=0)
    normalized = (features - columnMean) / columnStd
    return list(normalized)

if __name__ == '__main__':
    data = load_data('countries.csv')
    print(np.shape(data))
    countryNames = [row['Country'] for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)
    n = 204
    Z_raw = hac(features_normalized[:n])
    fig = fig_hac(Z_raw, countryNames[:n])
    plt.show()
