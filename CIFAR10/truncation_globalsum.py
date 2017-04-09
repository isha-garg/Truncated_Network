import caffe
import numpy as np


class GlobalSumLayer(caffe.Layer):
  def setup(self, bottom, top):
    pass

  def reshape(self, bottom, top):
    [batch_sz,depth2,filter_w,filter_h]=np.shape(bottom[0])
    top[0].reshape(batch_sz,depth2)
    #pass

  def forward(self, bottom, top):
    int_1=np.sum(bottom[0].data,-1)
    top[0].data[...]=np.sum(int_1,-1)

  def backward(self, top, propagate_down, bottom):
    pass
