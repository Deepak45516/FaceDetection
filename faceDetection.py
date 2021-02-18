#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


face_cascade = cv2.CascadeClassifier('/home/doruk/Downloads/haarcascade_frontalface_default.xml')


# In[3]:


color_image= cv2.imread("/home/doruk/Downloads/Deepak photo.jpg")


# In[4]:


gray_image=cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)


# In[5]:


faces = face_cascade.detectMultiScale(gray_image,1.1,5)


# In[6]:


for (x,y,w,h) in faces:
    cv2.rectangle(color_image, (x,y), (x + w, y + h ), (0,255, 0), 4)


# In[7]:


cv2.imshow('Image', color_image)


# In[ ]:


cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:




