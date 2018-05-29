import os
import matplotlib
from matplotlib import pylab, mlab, pyplot
from IPython.core.pylabtools import figsize, getfigs
plt = pyplot
import numpy as np
from PIL import Image


import analysis_helpers as h
reload(h)
## get standardized object list
categories = ['bird','car','chair','dog']
obj_list = []
for cat in categories:
    for i,j in h.objcat.iteritems():
        if j==cat:
            obj_list.append(i)

if __name__ == "__main__":

    ## path to curated sketches for figure
    path_to_close = 'sketches_pilot2/curated/close'
    path_to_far = 'sketches_pilot2/curated/far'
    path_to_objects = 'sketches_pilot2/curated/objects'

    ## get full list of close and far paths
    close_paths = [os.path.join(path_to_close,'{}.png'.format(i)) for i in obj_list]
    far_paths = [os.path.join(path_to_far,'{}.png'.format(i)) for i in obj_list]
    obj_paths =  [os.path.join(path_to_objects,'{}.png'.format(i)) for i in obj_list]

    assert len(far_paths)==32
    assert len(close_paths)==32
    assert len(obj_paths)==32

    ## render out close gallery
    print('Rendering out close gallery...')
    fig = plt.figure(figsize=(8,16),frameon=False)
    for i,f in enumerate(close_paths):
        im = Image.open(f)
        p = plt.subplot(8,4,i+1)
        plt.imshow(im)
        k = p.get_xaxis().set_ticklabels([])
        k = p.get_yaxis().set_ticklabels([])
        k = p.get_xaxis().set_ticks([])
        k = p.get_yaxis().set_ticks([])
        p.axis('off')

    plt.savefig('./plots/close_gallery.pdf')
    plt.savefig('../manuscript/figures/raw/close_gallery.pdf')

    ## render out far gallery
    print('Now rendering out far gallery...')
    fig = plt.figure(figsize=(8,16),frameon=False)
    for i,f in enumerate(far_paths):
        im = Image.open(f)
        p = plt.subplot(8,4,i+1)
        plt.imshow(im)
        k = p.get_xaxis().set_ticklabels([])
        k = p.get_yaxis().set_ticklabels([])
        k = p.get_xaxis().set_ticks([])
        k = p.get_yaxis().set_ticks([])
        p.axis('off')

    plt.savefig('./plots/far_gallery.pdf')
    plt.savefig('../manuscript/figures/raw/far_gallery.pdf')

    ## render out object far_galleryfig = plt.figure(figsize=(8,16),frameon=False)
    print('Now rendering out object gallery...')
    remove_gray = False
    fig = plt.figure(figsize=(16,32),frameon=False)
    for i,f in enumerate(obj_paths):
        im = Image.open(f)
        if remove_gray == True:
            im = im.convert("RGBA")
            pixdata = im.load()
            width, height = im.size
            for y in xrange(height):
                for x in xrange(width):
                    if pixdata[x, y] == (127, 127, 127, 255):
                        pixdata[x, y] = (127, 127, 127, 0)
            im.save(f, "PNG")
        p = plt.subplot(8,4,i+1)
        plt.imshow(im)
        k = p.get_xaxis().set_ticklabels([])
        k = p.get_yaxis().set_ticklabels([])
        k = p.get_xaxis().set_ticks([])
        k = p.get_yaxis().set_ticks([])
        p.axis('off')

    plt.savefig('./plots/object_gallery.pdf')
    plt.savefig('../manuscript/figures/raw/object_gallery.pdf')
