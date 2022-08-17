# %%
import numpy as np
import numpy.ma as ma
a = np.array([979., 636., 291., 587., 487., 300.])
mask = np.array([0,0,1,1,1,1])
maskarr = ma.masked_array(a, mask)
print(a)
print(mask)
print(maskarr)
print(maskarr.argmax())
# %%
