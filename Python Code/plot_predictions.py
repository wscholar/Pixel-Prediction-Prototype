from __future__ import division
import os
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, pyplot as plt

def plot_predictions(img_actual, img_predicted, img_error, root_dir, name, plot_vmin, plot_vmax, grayscale=False):
    print plot_vmin, plot_vmax
    plt.close()
    fig = plt.figure(figsize=(15,5.5))
    img_error /= (img_actual.max() - img_actual.min())
    ax1 = plt.subplot(131)
    ax1.set_title('Actual')
    colormap = cm.gray if grayscale else None
    im = plt.imshow(img_actual, interpolation='none', vmin=plot_vmin, vmax=plot_vmax, cmap=colormap)
    divider = make_axes_locatable(ax1)
    ax2 = plt.subplot(132)
    ax2.set_title('Predicted')
    im = plt.imshow(img_predicted, interpolation='none', vmin=plot_vmin, vmax=plot_vmax, cmap=colormap)
    divider = make_axes_locatable(ax2)
    ax3 = plt.subplot(133)
    ax3.set_title('Error')
    im = plt.imshow(img_error, interpolation='none', vmin=0., vmax=1.)  
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.subplots_adjust(wspace=0.1)
    if not os.path.exists(root_dir + "/pdfs"):
        os.makedirs(root_dir + "/pdfs")
    plt.savefig('%s/pdfs/%s.pdf' % (root_dir, name), bbox_inches="tight") 
    if not os.path.exists(root_dir + "/pngs"):
        os.makedirs(root_dir + "/pngs")    
    plt.savefig('%s/pngs/%s.png' % (root_dir, name), dpi=300, bbox_inches="tight")

def plot_interpolations(frame, root_dir, name, plot_vmin, plot_vmax, grayscale=False):
    print plot_vmin, plot_vmax
    plt.close()
    fig = plt.figure(figsize=(15,5.5))
    ax1 = plt.gca()
    ax1.set_title('Interpolated Video')
    colormap = cm.gray if grayscale else None
    im = plt.imshow(frame, interpolation='none', vmin=plot_vmin, vmax=plot_vmax, cmap=colormap)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.subplots_adjust(wspace=0.1)
    if not os.path.exists(root_dir + "/pdfs"):
        os.makedirs(root_dir + "/pdfs")
    plt.savefig('%s/pdfs/%s.pdf' % (root_dir, name), bbox_inches="tight") 
    if not os.path.exists(root_dir + "/pngs"):
        os.makedirs(root_dir + "/pngs")    
    plt.savefig('%s/pngs/%s.png' % (root_dir, name), dpi=300, bbox_inches="tight") 
