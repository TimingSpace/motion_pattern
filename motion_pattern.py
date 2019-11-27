from sklearn.decomposition import PCA
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import transformation as tf
data = np.loadtxt(sys.argv[1])
data = data[0:1200,:]
data = tf.pos_quats2SEs(data)
data = tf.pose2motion(data)
data = tf.SEs2ses(data)

pca = PCA(n_components=3)
pca.fit(data[:,0:3]) 
data_tranformed = pca.transform(data[:,0:3])

ax = plt.subplot(321)
ax.plot(data_tranformed[:,0],label='1st axis')
ax.plot(data_tranformed[:,1],label='2st axis')
ax.plot(data_tranformed[:,2],label='3st axis')
ax.legend(loc='upper right', shadow=True)
ax.text(0.0, 0.1, "Translation after PCA", fontsize=14,transform=ax.transAxes)

ax = plt.subplot(322, projection='3d')
#ax.plot(pca.components_[:0],pca.components_[:,1],pca.components_[:,2])
for i in range(0,3):
    ax.plot([0,pca.components_[i,0]*pca.explained_variance_ratio_[i]],\
        [0,pca.components_[i,1]*pca.explained_variance_ratio_[i]],\
        [0,pca.components_[i,2]*pca.explained_variance_ratio_[i]])
ax.set_xlabel('X ')
ax.set_ylabel('Y ')
ax.set_zlabel('Z ')


ax = plt.subplot(325)
ax.hist(data_tranformed[:,0])

pca.fit(data[:,3:6]) 
data_tranformed = pca.transform(data[:,3:6])

ax = plt.subplot(323)
ax.plot(data_tranformed[:,0],label='1st axis')
ax.plot(data_tranformed[:,1],label='2st axis')
ax.plot(data_tranformed[:,2],label='3st axis')
ax.legend(loc='upper right', shadow=True)
ax.text(0.0, 0.1, "Rotation after PCA", fontsize=14,transform=ax.transAxes)

ax = plt.subplot(324, projection='3d')
for i in range(0,3):
    ax.plot([0,pca.components_[i,0]*pca.explained_variance_ratio_[i]],\
        [0,pca.components_[i,1]*pca.explained_variance_ratio_[i]],\
        [0,pca.components_[i,2]*pca.explained_variance_ratio_[i]])
ax.set_xlabel('X ')
ax.set_ylabel('Y ')
ax.set_zlabel('Z ')


ax = plt.subplot(326)
ax.hist(data_tranformed[:,0])


plt.show()

print(pca.explained_variance_ratio_)
print(pca.singular_values_) 
print(pca.components_)
