The MultiConf Package
====================================

Package Wrapper
---------------

.. autoclass:: multiconf.icp_wrapper.ICPWrapper
   :members: fit,calibrate,predict

Inductive Conformal Predictor
-----------------------------

.. autoclass:: multiconf.icp_predictor.InductiveConformalPredictor
   :members: calibrate,predict

Prediction Regions
------------------

.. autoclass:: multiconf.prediction_regions.PredictionRegions
   :special-members: __call__
   :members: evaluate
