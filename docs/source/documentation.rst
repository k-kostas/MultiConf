The Structural Penalties ICP Package
====================================

Package Wrapper
---------------

.. autoclass:: structural_penalties_icp.icp_wrapper.ICPWrapper
   :members: fit,calibrate,predict

Inductive Conformal Predictor
-----------------------------

.. autoclass:: structural_penalties_icp.icp_predictor.InductiveConformalPredictor
   :members: calibrate,predict

Prediction Regions
------------------

.. autoclass:: structural_penalties_icp.prediction_regions.PredictionRegions
   :special-members: __call__
   :members: evaluate
