# Background
3D points cloud data is widely used in the industry to represent shapes in the real world. Compared with the 3D grids, format of point like (x, y, z) is more intuitive and compatible with mathematics. Not to mention that 3D points data is much easier to store and read. Deep learning tasks on 3d points data mainly focus on object classfication and segmentation. One popular model that is used currently to solve these tasks is the PointNet, which trained a permutation invariant model on set of points. However, a set of points could also be seen as a graph. In this project, we will explore the use of Graph Neural Network (GNN) on 3d Points cloud classfication and comapre its performance with the PointNet. 
![Image of background](https://raw.githubusercontent.com/ctwayen/Graph-Neural-Network-on-3D-Points/main/images/background.PNG)

# DataSet: ModelNet40 and ShapeNet
In this project, we will mainly train our model on ModelNet40 dataset. Collected by Princeton, ModelNet40 contains 40 categories of 3d shapes such as airplanes, cars, guitars, etc. Eash sample may have 1k - 60k points and also the surface information. Since surface data is not needed in this project, we had a simple pre-processing procedure that extract points from all samples and store them in h5py files. The points data will be directly used as the input for our PointNet model. 

Points data will also be used to constrauct graph. Each data samples (shapes) will be used to construct one corresponding graph using either k-nearest-neighbor or fix-raidus. 


# How do we construct the graph?
Like we mentioned, we used k-nearest-neighbor and fix-radius to consturct our graph from points.

* fix-radius:

    
![Image of background](https://raw.githubusercontent.com/ctwayen/Graph-Neural-Network-on-3D-Points/main/images/dataset.PNG)

# Mothods: models with permutation invariancy

# Our results:

## Helpful Links:

### Wanna replicate our results? Try following command: