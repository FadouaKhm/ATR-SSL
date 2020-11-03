import numpy as np
from astropy.stats import median_absolute_deviation

    
def invert(frame):

    return ~frame

def clip1(frame,threshold):
    
    lims = np.percentile(frame,threshold)
    clipped_frame = np.maximum(frame, lims[0])
    clipped_frame = np.minimum(lims[1], clipped_frame)
    clipped_frame = (clipped_frame - clipped_frame.min())/( clipped_frame.max() - clipped_frame.min())*255

    return clipped_frame



def clip2(frame,lims):

    clipped_frame = np.maximum(frame, lims[0])
    clipped_frame = np.minimum(lims[1], clipped_frame)
    clipped_frame = (clipped_frame - clipped_frame.min())/( clipped_frame.max() - clipped_frame.min())*255
    return clipped_frame
        


def normalize(frame,opts,params):
    if opts == 1:
        output = (frame-np.mean(frame))/np.std(frame)
    
    if opts == 2:
        output = (frame - params['mean'])/params['std']
    
    if opts == 3:
        output = (frame-np.median(frame))/median_absolute_deviation(frame)
    
    if opts == 4:
        output = (frame - params['median'])/params['mad']
        
    return output



def pre_process(frame, bool_invert, clip1_opts, normalize_opts, clip2_opts):
    if bool_invert == 1:
        frame = invert(frame)
    
    if clip1_opts['select'] == 1:
        frame = clip1(frame,np.array([clip1_opts["lower_threshold"], clip1_opts["upper_threshold"]]))
    
    if normalize_opts['select'] == 1:
        frame = normalize(frame,normalize_opts['technique'] ,dict())
    
    if clip2_opts['select'] == 1:
        frame = clip2(frame,np.array([clip2_opts["lower_threshold"], clip2_opts["upper_threshold"]]))
                
    return frame
            
        
    
    
