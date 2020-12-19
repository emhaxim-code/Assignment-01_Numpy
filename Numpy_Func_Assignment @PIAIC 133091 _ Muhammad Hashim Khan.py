#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 

print("Numpy_Func # 01 : array")
 
arr=np.array([1,2,3])  
print(arr)  


# In[2]:



print("Numpy_Func # 02 : concatenate ")
x=np.array([[1,2],[3,4]])  
y=np.array([[12,30]])  
z=np.concatenate((x,y))  
z  


# In[2]:


import numpy as np 

print("Numpy_Func # 03 : append() ")

a=np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])  
b=np.array([[11, 21, 31], [42, 52, 62], [73, 83, 93]])  
c=np.append(a,b)  
c  


# In[ ]:





# In[4]:


print("Numpy_Func # 04 : reshape() ") 
    
x=np.arange(12)  
y=np.reshape(x, (4,3))  
x  
y  


# In[6]:


print("Numpy_Func # 05 : sum() ") 

import numpy as np  
a=np.array([0.4,0.5])  
b=np.sum(a)  
b


# In[13]:


print("Numpy_Func # 06 : random() ") 

a=np.random.randn(2,2)  
a  


# In[14]:


print("Numpy_Func # 07 : zeros() ") 
  
a=np.zeros(6)  
a  


# In[15]:


print("Numpy_Func # 08 : np() ") 

a=np.array([2, 4, 6, 3**8])  
a  
b=np.log(a)  
b  
c=np.log2(a)  
c  
d=np.log10(a)  
d  


# In[ ]:


print("Numpy_Func # 09 : np() ") 

a=np.array([2, 4, 6, 3**8])  
a  
b=np.log(a)  
b  
c=np.log2(a)  
c  
d=np.log10(a)  


# In[ ]:





# In[16]:


print("Numpy_Func # 10 : where() ")

a=np.arange(12)  
b=np.where(a<6,a,5*a)  
b  


# In[17]:


print("Numpy_Func # 11 : argsort() ")

a=np.array([456,11,63])  
a  
b=np.argsort(a)  
b 


# In[18]:


print("Numpy_Func # 12 : transpose() ")

a= np.arange(6).reshape((2,3))  
a  
b=np.transpose(a)  
b  


# In[19]:


print("Numpy_Func # 13 : mean() ")

a = np.array([[1, 2], [3, 4]])  
b=np.mean(a)  
b  
x = np.array([[5, 6], [7, 34]])  
y=np.mean(x)  
y


# In[21]:


print("Numpy_Func # 14 : unique() ")

a=np.unique([1,2,3,4,3,6,2,4])  
a  


# In[26]:


print("Numpy_Func # 15 : tolist() ")

a = np.uint32([6, 2])  
a   
a_list=list(a)  
a_list  
type(a_list[0])  
a_tolist=a.tolist()  
a_tolist  
type(a_tolist[0])  


# In[2]:


import numpy as np

print("Numpy_Func # 16 : dot() ")

a = [[1, 2], [4, 1]]  
b = [[4, 11], [2, 3]]  
c=np.dot(a, b)  
c  


# In[ ]:





# In[3]:


print("Numpy_Func # 17 : loadtxt() ")

from io import StringIO  
c = StringIO(u"0 1\n2 3")  
c  
np.loadtxt(c) 


# In[6]:


print("Numpy_Func # 18 : clip() ")

a = np.arange(12)  
a
np.clip(a, 3, 9, out=a)  
a 


# In[2]:


print("Numpy_Func # 19 : flatten() ")

import numpy as np

a = np.array([[1,4,7], [2,5,8],[3,6,9]])  
b=a.flatten()  
b  


# In[3]:


print("Numpy_Func # 20 : meshgrid() ")


import numpy as np  
na, nb = (5, 3)  
a = np.linspace(1, 2, na)  
b = np.linspace(1, 2, nb)  
xa, xb = np.meshgrid(a, b)  
xa  
xb  


# In[4]:


print("Numpy_Func # 21 : std() ")

a=np.array([[1,4,7,10],[2,5,8,11]])  
b=np.std(a)  
b


# In[6]:


print("Numpy_Func # 22 : argmax() ")


 
x = np.arange(20).reshape(4,5) + 7  
x  
y=np.argmax(a)  
y  


# In[7]:


print("Numpy_Func # 23 : diff() ")

arr = np.array([0, 1, 2], dtype=np.uint8)  
arr  
b=np.diff(arr)  
b  
arr[2,...] - arr[1,...] - arr[0,...]


# In[8]:


print("Numpy_Func # 24 : empty() ")

x = np.empty([3, 2])  
x  


# In[9]:


print("Numpy_Func # 25 : histogram() ")


a=np.histogram([1, 5, 2], bins=[0, 1, 2, 3])  
a  


# In[ ]:


print("Numpy_Func # 26 : sort() ")

x=np.array([[1,4,2,3],[9,13,61,1],[43,24,88,22]])  
x  
y=np.sort(x)  
y  


# In[ ]:


print("Numpy_Func # 27 : average() ")

data = list(range(1,6))  
output=np.average(data)  
data  
output  


# In[ ]:


print("Numpy_Func # 28 : pad() ")

 
x = [1, 3, 2, 5, 4]  
y = np.pad(x, (3, 2), 'constant', constant_values=(6, 4))  
y 


# In[10]:


print("Numpy_Func # 29 : ravel() ")

x = np.array([[1, 3, 5], [11, 35, 56]])  
y = np.ravel(x, order='F')  
z = np.ravel(x, order='C')  
p = np.ravel(x, order='A')  
q = np.ravel(x, order='K')  
y  
z  
p  
q  


# In[12]:


print("Numpy_Func # 30 : arrcos() ")

arr = [0, 0.3, -1]   
print ("Input array : \n", arr)   
  
arccos_val = np.arccos(arr)   
print ("\nInverse cos values : \n", arccos_val) 


# In[13]:


print("Numpy_Func # 31 : srcsin() ")

import math  
  
arr = [0, 0.3, -1]   
print ("Input array : \n", arr)   
  
arcsine = np.arcsin(arr)   
print ("\nInverse Sine values : \n", arcsine)   


# In[14]:


print("Numpy_Func # 32 : arctan() ")

arr = [0, 0.3, -1]   
print ("Input array : \n", arr)   
  
arctan_val = np.arctan(arr)   
print ("\nInverse tan values : \n", arctan_val)


# In[15]:


print("Numpy_Func # 33 : degrees() ")

import math  
  
arr = [0, math.pi/2, math.pi/4, math.pi/6 ]   
print ("Input array : \n", arr)   
  
degval = np.degrees(arr)   
print ("\n Degree value : \n", degval)  


# In[16]:


print("Numpy_Func # 34 : tan() ")

import math  

arr = np.array([0, math.pi/4, 3*math.pi/2, math.pi/6])  
print("Input Array:",arr)  
print("tan Array:",end=" ")  
tanarr = np.tan(arr)  
print(tanarr)


# In[17]:


print("Numpy_Func # 35 : deg2rad() ")

import math  
  
arr = [0, math.pi/2, math.pi/4, math.pi/6 ]   
print ("Input array : \n", arr)   
  
radval = np.deg2rad(arr)   
print ("\n Radian value : \n", radval)


# In[18]:


print("Numpy_Func # 36 : hypot() ")

base = [10,2,5,50]  
per= [3,10,23,6]  
  
print("Input base array:",base)  
print("Input perpendicular array:",per)  
  
hyp = np.hypot(base,per)  
  
print("hypotenuse ",hyp)


# In[19]:


print("Numpy_Func # 37 : rad2deg() ")

import math  
  
arr = [0, math.pi/2, math.pi/4, math.pi/6 ]   
print ("Input array : \n", arr)   
  
degval = np.rad2deg(arr)   
print ("\n Degree value : \n", degval) 


# In[20]:


print("Numpy_Func # 38 : radians() ")

arr = [0, 30, 60, 90 ]   
print ("Input array : \n", arr)   
  
radval = np.radians(arr)   
print ("\n Radian value : \n", radval) 


# In[21]:


print("Numpy_Func # 39 : arcsinh() ")

import math  
  
arr = np.array([0, math.pi/4, 3*math.pi/2, math.pi/6])  
  
print("Input Array:",arr)  
print("tanh Array:",end=" ")  
  
arcsinharr = np.arcsinh(arr)  
  
print(arcsinharr)


# In[22]:


print("Numpy_Func # 40 : arctanh() ")

import math  
  
arr = np.array([0,0.2, 0.5, 0.3])  
  
print("Input Array:",arr)  
print("arctanh Array:",end=" ")  
  
arctanharr = np.arctanh(arr)  
  
print(arctanharr) 


# In[23]:


print("Numpy_Func # 41 : degrees() ")

import math  
  
arr = [0, math.pi/2, math.pi/4, math.pi/6 ]   
print ("Input array : \n", arr)   
  
degval = np.degrees(arr)   
print ("\n Degree value : \n", degval)   


# In[ ]:


print("Numpy_Func # 43 : ravel() ")


# In[24]:


print("Numpy_Func # 42 : tan() ")

import math  
arr = np.array([0, math.pi/4, 3*math.pi/2, math.pi/6])  
print("Input Array:",arr)  
print("tan Array:",end=" ")  
tanarr = np.tan(arr)  
print(tanarr)  


# In[25]:


print("Numpy_Func # 42 : deg2rad() ")

import math  
  
arr = [0, math.pi/2, math.pi/4, math.pi/6 ]   
print ("Input array : \n", arr)   
  
radval = np.deg2rad(arr)   
print ("\n Radian value : \n", radval)  


# In[26]:


print("Numpy_Func # 43 : hypot() ")

base = [10,2,5,50]  
per= [3,10,23,6]  
  
print("Input base array:",base)  
print("Input perpendicular array:",per)  
  
hyp = np.hypot(base,per)  
  
print("hypotenuse ",hyp) 


# In[ ]:


print("Numpy_Func # 44 : rad2deg() ")

import math  
  
arr = [0, math.pi/2, math.pi/4, math.pi/6 ]   
print ("Input array : \n", arr)   
  
degval = np.rad2deg(arr)   
print ("\n Degree value : \n", degval) 


# In[27]:


print("Numpy_Func # 45 : radions() ")

arr = [0, 30, 60, 90 ]   
print ("Input array : \n", arr)   
  
radval = np.radians(arr)   
print ("\n Radian value : \n", radval)   


# In[28]:


print("Numpy_Func # 46 : ceil() ")

arr = [0.23, 0.09, 1.2, 1.24, 9.99]  
  
print("Input array:",arr)  
  
r_arr = np.ceil(arr)  
  
print("Output array:",r_arr)  
  
arr2 = [145.23, 0.12, 12.34, 123]  
  
r_arr2=np.ceil(arr2)  
  
print("Input array:",arr2)  
  
print("Output array:",r_arr2)


# In[29]:


print("Numpy_Func # 47 : floor() ")

arr = [0.23, 0.09, 1.2, 1.24, 9.99]  
  
print("Input array:",arr)  
  
r_arr = np.floor(arr)  
  
print("Output array:",r_arr)  
  
arr2 = [145.23, 0.12, 12.34, 123]  
  
r_arr2=np.floor(arr2)  
  
print("Input array:",arr2)  
  
print("Output array:",r_arr2)  


# In[30]:


print("Numpy_Func # 48 : arrang() ")

arr = np.arange(0,10,2,float)  
print(arr)


# In[31]:


print("Numpy_Func # 49 : asarray() ")

l=[[1,2,3,4,5,6,7],[8,9]]  
a = np.asarray(l);  
print(type(a))  
print(a)  


# In[32]:


print("Numpy_Func # 50 : linespace() ")

arr = np.linspace(10, 20, 5)  
print("The array over the given range is ",arr)  

