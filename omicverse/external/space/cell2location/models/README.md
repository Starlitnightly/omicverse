### The models are defined as python classes with the following inheritance structure to allow methods to be reused:

'- BaseModel - Methods needed for Pymc3, Pytorch and pyro implementations

'-- Pymc3Model - Methods for Pymc3 models

'--- Pymc3LocModel - Methods for Pymc3 location models: fixed gene weights (2nd dimension), estimating weights for observations (1st dimension).

'---- LocationModelLinearDependentW - Main cell2location model that accounts for similarity in locations of cell types.

'---- LocationModelLinearDependentWMultiExperiment - Extention of the main cell2location model to joint modelling of multiple spatial experiments.

'---- LocationModel - Simplified model that models cell type locations as independent.

  

'-- CoLocatedCombination_sklearnNMF - Class that provides methods for analysing the cell type abundance estimated by cell2location using NMF.

'-- ArchetypalAnalysis - Class that provides methods for analysing the cell type abundance estimated by cell2location using Archetypal Analysis.

  
  

'-- TorchModel - Methods for pytorch models

'--- RegressionTorchModel - Methods for pytorch regression models: fixed weights for observations (1st dimension), estimating gene weights (2nd dimension).

'---- RegressionGeneBackgroundCoverageTorch

'---- RegressionGeneBackgroundCoverageGeneTechnologyTorch

(In development)

'-- PyroModel - Methods for pyro models

'--- PyroLocModel - Methods for pyro location models: fixed gene weights (2nd dimension), estimating weights for observations (1st dimension).

'---- LocationModelLinearDependentWPyro - Pyro translation of the main cell2location model that accounts for similarity in locations of cell types.

'---- LocationModelPyro - Pyro translation of the simplified model.
