# PartialJac

Implementation of Critical Initialization of Wide and Deep Neural Networks through Partial Jacobians: General Theory and Applications (https://arxiv.org/abs/2111.12143)


Use the GradHook in utils/partialjaclib.py if one only wants to compute APJN. 
See resnet_phase_diagram.py for using it to plot phase diagram.

The implementation used for getting APJN for FCN and ViT is more tedious, but it records more intermediate informations.