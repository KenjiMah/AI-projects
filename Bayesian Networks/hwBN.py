

# Starting with defining the network structure
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination



#!!!!!!!!!!!!!!!  VERY IMPORTANT  !!!!!!!!!!!!!!!
# MAKE SURE to use the terms "MaryCalls", "JohnCalls", "Alarm",
# "Burglary" and "Earthquake" as the states/nodes of the Network.
# And also use "burglary_model" as the name of your Bayesian model.
########-----YOUR CODE STARTS HERE-----########
HW_model = BayesianModel([('P', 'C'), 
                          ('C', 'M'),
                          ('C', 'T')])

cpd_P = TabularCPD(variable='P', variable_card=2,
                      values=[[0.8], [0.2]])
cpd_C = TabularCPD(variable='C', variable_card=2,
                       values=[[.667, .15],
                               [.333, .85]],
                    evidence=['P'],
                    evidence_card=[2])
cpd_M = TabularCPD(variable='M', variable_card=2,
                       values=[[.6, .7],
                               [.4, .3]],
                    evidence=['C'],
                    evidence_card=[2])
cpd_T = TabularCPD(variable='T', variable_card=2,
                       values=[[.5, .25],
                               [.5, .75]],
                    evidence=['C'],
                    evidence_card=[2])

HW_model.add_cpds(cpd_P,cpd_C,cpd_M,cpd_T)
########-----YOUR CODE ENDS HERE-----########

# Doing exact inference using Variable Elimination
HW_infer = VariableElimination(HW_model)

########-----YOUR MAY TEST YOUR CODE BELOW -----########
########-----ADDITIONAL CODE STARTS HERE-----########

   
print(HW_infer.query(variables=['M'], evidence={'C': 1}))
print(HW_infer.query(variables=['M'], evidence={'C': 1, 'P':1}))
print(HW_infer.query(variables=['M']))
print(HW_infer.query(variables=['T'], evidence={'M':1}))
########-----YOUR CODE ENDS HERE-----########


