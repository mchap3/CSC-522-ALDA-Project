'''
Final interface of code. Central place to view/run all experiments, run final optimized models, and summarize results.
Please note that the results observed may not match those reported, as many of the models are stochastic and generate
a spread of results. The results we reported were close to median performance for each model.
'''

import visualization
import datamanipulation
import indicators
import experiments
import accountperformance

# I) This first section contains global variable definitions and initializes everything that is needed for the rest of
# the code to run.


# II) This section of code allows the experiments to be run and examined. Feel free to uncomment any of the
# function calls and see the results.

# I.A) KNN Experiments
# experiments.KNN_neighbor_optimization()

# I.B) NB Experiments


# I.C) RF Experiments
# experiments.RF_optimization()
# experiments.RF_comparison()

# I.D) ANN Experiments
# experiments.ANN_optimization()
# experiments.ANN_regularization()

# I.E) SVM Experiments


# III) This second section of code runs our optimized models, generates predictions for our test set, and creates summary
# tables/plots showing the run results.

