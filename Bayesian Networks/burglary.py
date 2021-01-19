###################################################
##             ASU CSE 571 ONLINE                ##
##        Unit 3 Reasoning under Uncertainty     ##
##             Project Submission File           ##
##                 burglary.py                   ##
###################################################

###################################################
##                !!!IMPORTANT!!!                ##
##        This file will be auto-graded          ##
##    Do NOT change this file other than at the  ##
##       Designated places with your code        ##
##                                               ##
##  READ the instructions provided in the code   ##
###################################################

# Starting with defining the network structure
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def buildBN():

    #!!!!!!!!!!!!!!!  VERY IMPORTANT  !!!!!!!!!!!!!!!
    # MAKE SURE to use the terms "MaryCalls", "JohnCalls", "Alarm",
    # "Burglary" and "Earthquake" as the states/nodes of the Network.
    # And also use "burglary_model" as the name of your Bayesian model.
    ########-----YOUR CODE STARTS HERE-----########
    burglary_model = BayesianModel([('Burglary', 'Alarm'), 
                              ('Earthquake', 'Alarm'),
                              ('Alarm', 'JohnCalls'),
                              ('Alarm', 'MaryCalls')])

    cpd_burg = TabularCPD(variable='Burglary', variable_card=2,
                          values=[[0.999], [0.001]])
    cpd_earth = TabularCPD(variable='Earthquake', variable_card=2,
                           values=[[0.998], [0.002]])
    cpd_alarm = TabularCPD(variable='Alarm', variable_card=2,
                           values=[[.999, .06, .71, .05],
                                   [.001, .94, .29, .95]],
                        evidence=['Earthquake', 'Burglary'],
                        evidence_card=[2, 2])
    cpd_johnCalls = TabularCPD(variable='JohnCalls', variable_card=2,
                               values=[[.95, .1], [.05, .9]],
                               evidence=['Alarm'], evidence_card=[2])
    cpd_maryCalls = TabularCPD(variable='MaryCalls', variable_card=2,
                               values=[[.99, .3], [.01,.7]],
                               evidence=['Alarm'], evidence_card=[2])
    
    burglary_model.add_cpds(cpd_burg, cpd_earth, cpd_alarm, cpd_johnCalls, cpd_maryCalls)
    ########-----YOUR CODE ENDS HERE-----########
    
    # Doing exact inference using Variable Elimination
    burglary_infer = VariableElimination(burglary_model)

    ########-----YOUR MAY TEST YOUR CODE BELOW -----########
    ########-----ADDITIONAL CODE STARTS HERE-----########

    print(burglary_infer.query(['JohnCalls'], evidence={'Earthquake': 0}))
    print(burglary_infer.query(['MaryCalls'], evidence={'Earthquake': 0, 'Burglary':1}))
    print(burglary_infer.query(['MaryCalls'], evidence={'Earthquake': 1, 'Burglary':1}))
    ########-----YOUR CODE ENDS HERE-----########
    
    return burglary_infer


buildBN()