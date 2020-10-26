#!/usr/bin/env python
# coding: utf-8

# In[1]:


import corner
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import json
import lmfit


# In[2]:


with open('flux for2.txt') as fr:
	fl1 = json.load(fr)
with open('flux for8.txt') as fr:
	fl2 = json.load(fr)
with open('time.txt') as fr:
	time = json.load(fr)
with open('position of core for2.txt') as fr:
	position1 = json.load(fr)
with open('position of core for8.txt') as fr:
	position2 = json.load(fr)


# In[3]:


crsh = np.array(position1) - np.array(position2)

inaccuracy_crshf=crsh*0.1

fl1=np.array(fl1)
fl2=np.array(fl2)


# In[4]:


flux = np.array([fl1, fl2])
position1=np.array(position1)


# In[5]:


def func(flux,a, b1, k, b2):
	return (a+b1*(flux[0]**k) - b2*(flux[1]**k))


# In[6]:


#p0 = (0, 0.1, 0.23,0.1)

popt, pcov = curve_fit(func, flux, crsh,maxfev=10000,sigma=inaccuracy_crshf, bounds =([0., 0.9, 0.2, 0.9 ],[0.2, 1.5, 0.26,  1.5 ]))

perr = np.sqrt(np.diag(pcov))
print( popt)
print(pcov)
print("deviation", perr)


# In[7]:


ans = popt[0] + popt[1]*pow(flux[0],popt[2]) - popt[3]*pow(flux[1],popt[2])


# In[8]:


plt.plot (time, ans, color = 'red', label = "optimized data")
plt.plot (time, crsh, color = 'blue', label = "data")


plt.legend ()


plt.xlabel('time')
plt.ylabel('coreshift')
plt.show()


# In[9]:


plt.plot (flux[0]**popt[2]-flux[1]**popt[2], crsh, 'o', color = 'red', label = "data")

plt.plot (flux[0]**popt[2]-flux[1]**popt[2], ans, '-', color = 'blue', label = "optimized data")
plt.legend ()

plt.xlabel('flux difference between 2 and 8.1 GHz')
plt.ylabel('coreshift')
plt.show()


# In[10]:


def func(x, a, b1, k,):
    return a+b1*(x**k)
x = fl2.copy()
y = np.array(position2)

model = lmfit.Model(func)


# In[11]:


p = model.make_params(a=0, b1=1, k=0.28)


# In[12]:


result = model.fit(data=y,x=fl1, params=p)
lmfit.report_fit(result)
result.plot()


# In[13]:


fl2_2 = []
for i in range(len(fl1)):
    if fl1[i] != max(fl1):
         fl2_2.append(fl1[i])
    else:
        break


# In[16]:


fl2_2 =  np.array(fl2_2)
y1  = y[0:fl2_2.shape[0]]
x = fl2_2.copy()
model = lmfit.Model(func)


# In[17]:


p = model.make_params(a=0, b1=1, k=0.28)
result = model.fit(data=y1,x=fl2_2, params=p)
lmfit.report_fit(result)
result.plot()


# In[ ]:




