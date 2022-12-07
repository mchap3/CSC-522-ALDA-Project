'''
Final interface of code. Central place to view/run all experiments, run final optimized models, and summarize results.
Please note that the results observed may not match those reported, as many of the models are stochastic and generate
a spread of results. The results we reported were close to median performance for each model.

The default code produces outputs for each model with and without indicators and summarizes all of them into summary
tables/plots. To view our experiments, comment out the portion of the code that produces the result summaries (section
3), and uncomment individual functions from section 2 of this script.

Wish we could have made a better user interface, but that was beyond the scope of the project! Enjoy ^_^
'''

import visualization
import SVM
import NaiveBayes

""" 
I) This first section allows you to choose what results you would like to see! Takes some user inputs and produces the
appropriate output.
"""
ans = input('What would you like to see? Please note that ideal values overshadow most model results if plotted. \n'
            'A) Model result summaries with indicators, ideal values plotted.\n'
            'B) Model result summaries with indicators, ideal values NOT plotted.\n'
            'C) Model result summaries without indicators, ideal values plotted.\n'
            'D) Model result summaries without indicators, ideal values NOT plotted.\n')

ans = ans.upper()
if ans == 'A':
    visualization.project_summary(plotideal=True)
elif ans == 'B':
    visualization.project_summary()
elif ans == 'C':
    visualization.project_summary(add_indicators=False, plotideal=True)
elif ans == 'D':
    visualization.project_summary(add_indicators=False)
else:
    print('Sorry, you did not provide a valid input. Please try again.')


"""
II) This section of code allows the experiments to be run and examined. Feel free to uncomment any of the
function calls and see the results.
"""

# I.A) KNN Experiments
# experiments.KNN_neighbor_optimization()

# I.B) NB Experiments
# NaiveBayes.run_NB_optimization_experiment()
# NaiveBayes.run_NB_final_experiment()

# I.C) RF Experiments
# experiments.RF_optimization()
# experiments.RF_comparison()

# I.D) ANN Experiments
# experiments.ANN_optimization()
# experiments.ANN_regularization()

# I.E) SVM Experiments
# SVM.run_SVM_optimization_experiment()
# SVM.run_SVM_final_experiment()


