import imageio
import sknw
import networkx as nx
import numpy as np
import math
# from shapely.geometry import Polygon
import pickle
import pandas as pd
import statannot

import skimage
from scipy import ndimage

import matplotlib.pyplot as plt

# more props
# circularity = 4 * pi * area / perimeter^2
# curvature = 1 / radius of curvature
# fourier descriptors


# img = imageio.imread('/localhome/asa420/MIAL/data/confocal-data/vess_enh_unet/atl/gt_skel/atl1_proc_skel.png')

# invskel = np.invert(img)

# edt = ndimage.distance_transform_edt(invskel)

# cc = skimage.morphology.label(edt, connectivity=2)

# cntrs = skimage.measure.find_contours(cc, 0.8)

# # print(len(cntrs))

# plt.imshow(cc, cmap='gray')
# for cntr in cntrs:
#     plt.plot(cntr[:, 1], cntr[:, 0], linewidth=2)

# # plt.imshow(img, cmap='gray')
# plt.axis('off')
# plt.show()

# exit()


def get_shape_props(group):

    group_num_dict = {'atl': 26, 'climp': 31, 'control': 31, 'rtn': 29}

    df = pd.DataFrame()

    for num in range(1, group_num_dict[group] + 1):
        skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal-data/vess_enh_unet/{group}/gt_skel/{group}{num}_proc_skel.png')

        invskel = np.invert(skel)

        edt = ndimage.distance_transform_edt(invskel)

        cc = skimage.morphology.label(edt, connectivity=2)

        props = skimage.measure.regionprops_table(cc, properties=('area', 'area_convex', 'area_filled', 'axis_major_length', 'axis_minor_length', 'eccentricity', 'equivalent_diameter_area', 'feret_diameter_max', 'inertia_tensor', 'inertia_tensor_eigvals', 'orientation', 'perimeter'))

        new_df = pd.DataFrame.from_dict(props)

        # remove small objects
        new_df = new_df[(new_df['area'] > 3) & (new_df['area'] < 500)]

        df = pd.concat([df, new_df], ignore_index=True)

    # print(df)
        
    # df.to_csv(f'{group}_skel_props.csv', index=False)
    return df


# atl = get_shape_props('atl')
# climp = get_shape_props('climp')
# control = get_shape_props('control')
# rtn = get_shape_props('rtn')

# ndf = pd.concat([atl, climp, control, rtn], ignore_index=True)

# ndf.to_csv('all_skel_props.csv', index=False)

# data = pd.read_csv('skel_props_v2.csv')

# # remove the rows where area_convex is > 300
# data = data[data['area_convex'] < 300]

# data = data[data['area'] > 5]

# data.to_csv('skel_props_v2.csv', index=False)

df = pd.read_csv('skel_props_v3.csv')

df = df.drop(['id'], axis=1)

X = df.drop('target', axis=1)
y = df['target']

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from xgboost import XGBClassifier
import time

k_best_features = SelectKBest(score_func=f_classif, k=16)
X_selected = k_best_features.fit_transform(X, y)
selected_features = X.columns[k_best_features.get_support(indices=True)]
# print("Selected features:", selected_features)

# X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
# clf = RandomForestClassifier(random_state=42)
# clf.fit(X_train, y_train)

xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
training_start = time.perf_counter()
xgb.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = xgb.predict(X_test)
prediction_end = time.perf_counter()
acc_xgb = (preds == y_test).sum().astype(float) / len(preds)*100
xgb_train_time = training_end-training_start
xgb_prediction_time = prediction_end-prediction_start
print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))
print("Time consumed for training: %4.3f" % (xgb_train_time))
print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time))


# Evaluate classifier
# y_pred = clf.predict(X_test)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# Step 6: Interpretation and Validation
# Analyze feature importance provided by the classifier
# feature_importance = pd.Series(clf.feature_importances_, index=selected_features)
# print("Feature Importance:")
# print(feature_importance)




exit()

def calculate_angle(point, reference_point):
    x, y = point[0] - reference_point[0], point[1] - reference_point[1]
    return math.atan2(y, x)

def sort_points_clockwise(points):
    # Find the centroid as the reference point
    reference_point = [sum(x[0] for x in points) / len(points),
                       sum(x[1] for x in points) / len(points)]

    # Sort the points based on the angle with respect to the reference point
    sorted_points = sorted(points, key=lambda p: calculate_angle(p, reference_point))

    return sorted_points


def get_cycle_area(group, seq_num):
    area_data = []
    for frame in range(1, seq_num + 1):
        skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal-data/vess_enh_unet/{group}/skel/{group}{frame}_proc_skel.png')

        # build graph from skeleton
        graph = sknw.build_sknw(skel, multi=False, iso=False)

        # get the minimum cycle basis of the graph.
        min_cycles = nx.minimum_cycle_basis(graph.to_undirected())

        node_set = graph.nodes

        node_coords = np.array([node_set[node]['o'] for node in node_set])

        for cycle in min_cycles:
            # shoelace formula needs the points to be in clockwise order

            sorted_points = sort_points_clockwise(node_coords[cycle])

            xl = [point[0] for point in sorted_points]
            yl = [point[1] for point in sorted_points]
            
            if len(xl) > 3:
                polygon = Polygon(zip(xl, yl))
                area_data.append(polygon.area)

    return area_data


# atl_data = get_cycle_area('atl', 26)
# climp_data = get_cycle_area('climp', 31)
# control_data = get_cycle_area('control', 31)
# rtn_data = get_cycle_area('rtn', 29)

# with open('atl_cycles.pkl', 'wb') as f:
#     pickle.dump(atl_data, f)

# with open('climp_cycles.pkl', 'wb') as f:
#     pickle.dump(climp_data, f)

# with open('control_cycles.pkl', 'wb') as f:
#     pickle.dump(control_data, f)

# with open('rtn_cycles.pkl', 'wb') as f:
#     pickle.dump(rtn_data, f)


with open('atl_cycles.pkl', 'rb') as f:
    atl_data = pickle.load(f)


with open('climp_cycles.pkl', 'rb') as f:
    climp_data = pickle.load(f)

with open('control_cycles.pkl', 'rb') as f:
    control_data = pickle.load(f)

with open('rtn_cycles.pkl', 'rb') as f:
    rtn_data = pickle.load(f)

# plot boxplots for each group

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

df = pd.DataFrame()

df['Cycles_area'] = pd.Series(np.concatenate((control_data, rtn_data, climp_data, atl_data)))

df['Group'] = pd.Series(['Control']*len(control_data) + ['RTN']*len(rtn_data) + ['Climp']*len(climp_data) + ['ATL']*len(atl_data))

ax = sns.boxplot(x='Group', y='Cycles_area', data=df, showfliers=False)

yt = ax.get_yticks()
# ax.set_ylim([0.0, 800.])
# ax.set_yticklabels([f'{y:.2f}' for y in yt])
ax.set_yticklabels([f'{y}' for y in yt])

ax.set_xlabel('Group', fontsize=18)
ax.set_ylabel('Cycle Area', fontsize=18)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('ATL', 'Control'), ('Climp', 'RTN'), ('Climp', 'Control'), ('RTN', 'Control')]

# statannot.add_stat_annotation(ax, x='Group', y='Cycles_area', data=df, box_pairs=box_pairs,
                                #   test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize=20)
plt.gcf().set_size_inches(3.5, 10)
plt.savefig('cycle_area_boxplot.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.close()

# plt.show()




# data = [atl_data, climp_data, control_data, rtn_data]

# fig, ax = plt.subplots(figsize=(10, 10))
# ax.set_title('Cycle Area Distribution')
# ax.set_xlabel('Group')
# ax.set_ylabel('Cycle Area')
# ax.set_xticklabels(['ATL', 'Climp', 'Control', 'RTN'])


# ax.boxplot(data, showfliers=False)
# plt.show()
