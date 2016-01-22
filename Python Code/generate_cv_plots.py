from __future__ import division
import os
import subprocess
import numpy as np
from plot_predictions import plot_predictions
import pylab
import sys

def myround(x, prec=2, base=.05):
  return round(base * round(float(x)/base),prec)

def plot_all(root_dir, output_root, only_process=None, grayscale=None):
    for d in os.listdir(root_dir):
        if d.startswith("."):
            continue
        if d == str(only_process) or only_process is None:
            path = root_dir + d
            output_path = output_root + d
            max_int = -1
            for f in os.listdir(path):
                if f.startswith('.'): 
                    continue
                num_portion = int(f.split("_")[0])
                max_int = max(max_int, num_portion)
            ''' get dynamic range '''
            vmin = 1e1000
            vmax = -1e1000
            for i in range(1, max_int + 1):
                img_a = np.loadtxt("%s/%d_actual.txt" % (path,i))
                img_p = np.loadtxt("%s/%d_predicted.txt" % (path,i))
                combined = np.vstack((img_a, img_p))
                vmin = min(vmin, combined.min())
                vmax = max(vmax, combined.max())
            nvmin = myround(vmin)
            nvmax = myround(vmax)
            if nvmin < nvmax:
                vmin = nvmin
                vmax = nvmax
            ''' Output '''            
            for i in range(1, max_int + 1):
                img_actual = np.loadtxt("%s/%d_actual.txt" % (path,i))            
                img_predicted = np.loadtxt("%s/%d_predicted.txt" % (path,i))
                img_error = np.loadtxt("%s/%d_error.txt" % (path,i))
                plot_predictions(img_actual, img_predicted, img_error, output_path, 'chemist_prediction_%d' % i, vmin, vmax, grayscale)    
            ''' Generate gif '''
            gif_name = 'combined_%s.gif' % d
            #c = '"C:\\Program Files\\ImageMagick-6.8.9-Q8\\convert" -delay 40 -loop 0 pngs/*.png pngs/' + gif_name
            c = 'convert -delay 40 -loop 0 pngs/*.png pngs/' + gif_name
            p = subprocess.Popen(c, shell=True, cwd=output_path)
            p.wait()
            ''' Copy to top level gif directory '''
            gif_dir = output_root + "./gifs/"
            if not os.path.exists(gif_dir):
                os.makedirs(gif_dir)
            c = 'cp pngs/%s ../gifs/%s' % (gif_name, gif_name)
            p = subprocess.Popen(c, shell=True, cwd=output_path)
            p.wait()

if __name__ == "__main__":
    if (len(sys.argv) < 3):
        plot_all('../Results/trials-100-states_MXL_0/', '../images/trials-100-states_MXL_0/', grayscale=True)
    else:
        plot_all(sys.argv[1], sys.argv[2], grayscale=True)
    #plot_all('../Results/Test/Res/', '../images/Test/Res/', grayscale=True)
