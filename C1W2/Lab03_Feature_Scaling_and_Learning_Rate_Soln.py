import numpy as np
np.set_printoptions(precision=2)

import matplotlib.pyplot as plt
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'; 
plt.style.use("./C1W1/deeplearning.mplstyle")
from lab_utils_multi import  load_house_data, compute_cost, run_gradient_descent 
from lab_utils_multi import  norm_plot, plt_contour_multi, plt_equal_scale, plot_cost_i_w

X_train,y_train=load_house_data()
X_features=['size(sqft)','bedrooms','floors','age']

fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Prize(1000's")
plt.show()

for alpha_i in [9.9e-7,9e-7,1e-7]:
    _,_,hist=run_gradient_descent(X_train,y_train,10,alpha=alpha_i)
    plot_cost_i_w(X_train, y_train, hist)
    print(str(alpha_i))
