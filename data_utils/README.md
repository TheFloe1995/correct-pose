# Data Utils
This package contains classes and functions for anything related to the
data (loading, converting, transforming, accessing, storing, analyzing, 
visualizing etc.). 

## Datasets
The code expects the datasets to be provided in a special format:

* Pickle files ending on ".pt" that can be loaded with torch.load()
* A file can contain...
    * a single **float32** tensor of shape **(N, J, D)** per file (where N is the number of samples, 
      J is the number of joints and D is the number of dimensions),
    * a dictionary of tensors (type and shape like above), where each tensor represents a subset and
      all tensors have the exact same shape. This format is intended to allow convenient training 
      with the predictions of multiple backbone models on the same *base* dataset. It is not
      intended to be used for storing train/validation/test subsets into one file. 
* Input poses and labels are stored in different files, but they need to have the exact 
same number of samples and the name is constructed as `prefix_(poses|labels).pt`. 

## Pose Format
At the moment, arbitrary values for J and D are not supported by all parts of the code. 
Usually **J=21** and **D=3** is expected. They joint order in this format always follows the HANDS
2017 challenge convention:
  
    [Wrist,  
    TMCP, IMCP, MMCP, RMCP, PMCP,  
    TPIP, TDIP, TTIP,  
    IPIP, IDIP, ITIP,  
    MPIP, MDIP, MTIP,  
    RPIP, RDIP, RTIP,  
    PPIP, PDIP, PTIP],
          
where ’T’, ’I’, ’M’, ’R’, ’P’ denote ’Thumb’, ’Index’, ’Middle’, ’Ring’, ’Pinky’ fingers. 
’MCP’, ’PIP’, ’DIP’, ’TIP’ as in the following Figure:
![http://icvl.ee.ic.ac.uk/hands17](http://icvl.ee.ic.ac.uk/hands17/wp-content/uploads/sites/5/2017/06/hand_map-768x475.png
 "Hand Skeleton Definition")

## Other Remarks
**Distortions** refer to operations that leave the label unaffected but generate different poses.
**Augmentations** refer to operations that are applied to both the input pose and the label in 
order to generate a new sample pair.