from sklearn.decomposition import PCA
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import transformation as tf
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fig = plt.figure(figsize=(18,8))

#ax.set_xlim(-1000, 1000)
#ax.set_ylim(-500, 1500)
#plt.xticks(fontsize=25)
#plt.yticks(fontsize=25)
#plt.tight_layout()
#plt.tight_layout()

local_fs = 20
data = np.loadtxt(sys.argv[1])
#data = data[0:1200,:]

motion = tf.pos_quats2SEs(data) # uncommon this if the data is in quaternion format
#motion = data # uncommin this if data is rotation matrix 
motion = tf.pose2motion(motion)
motion = tf.SEs2ses(motion)
print(np.sum(motion,0))
print(motion)
pca = PCA(n_components=3)
pca.fit(motion[:,0:3]) 
data_tranformed = pca.transform(motion[:1200,0:3])
ax = fig.add_subplot(121)
ax.plot(data_tranformed[:,0],label='principle axis')
ax.plot(data_tranformed[:,1],label='secondary axis')
ax.plot(data_tranformed[:,2],label='third axis')
ax.legend(loc='upper right', shadow=True,fontsize=25)
ax.set_xlabel('time/frame',fontsize=local_fs)
ax.set_ylabel('translation/m',fontsize=local_fs)
ax.tick_params(length=1,width=1,labelsize=local_fs)
#ax.text(0.0, 0.1, "Translation after PCA", fontsize=14,transform=ax.transAxes)
t = pca.explained_variance_ratio_
print(pca.explained_variance_ratio_)
print(pca.singular_values_) 
print(pca.components_)

pca.fit(motion[:,3:6]) 
data_tranformed = pca.transform(motion[:1200,3:6])

ax = fig.add_subplot(122)
ax.plot(data_tranformed[:,0],label='principle axis')
ax.plot(data_tranformed[:,1],label='secondary axis')
ax.plot(data_tranformed[:,2],label='third axis')
ax.legend(loc='upper right', shadow=True,fontsize=25)
ax.set_xlabel('time/frame',fontsize=local_fs)
ax.set_ylabel('rotation/deg',fontsize=local_fs)
ax.tick_params(length=1,width=1,labelsize=local_fs)
#ax.text(0.0, 0.1, "Rotation after PCA", fontsize=14,transform=ax.transAxes)
r = pca.explained_variance_ratio_
print(pca.explained_variance_ratio_)
print(pca.singular_values_) 
print(pca.components_)

sigma = 0.5*(t[1]/t[0])*(t[2]/t[0])+0.5*(r[1]/r[0])*(r[2]/r[0])
print(sigma)

#plt.title('motion analysis of KITTI dataset')
plt.show()
