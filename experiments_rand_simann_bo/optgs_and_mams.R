# Install the package handling the Barber & Jennison optimization framework
install.packages("OptGS")
library(OptGS)

# Find optimal asymmetric trial configuration
# J=3 stages, alpha=0.025, power=0.9, delta=0.3
optimal_design <- optgs(
  J = 3, 
  alpha = 0.05, 
  power = 0.9, 
  delta1 = 1,
  sigma = 3,
  weights = c(0, 0, 1, 0) # Weight = 1 on minimizing ESS under Alternative Hypothesis
)

print(optimal_design)
plot(optimal_design)

# Groupsize:  65 
# Futility boundaries  0.15 1.1 1.79 
# Efficacy boundaries  2.14 1.91 1.79 
# ESS at null:     98.2 
# ESS at CRD:      109.7 
# Maximum ESS:     126 
# Max sample-size: 195 

install.packages("MAMS")
library(MAMS)

mams(K = 1, 
     J = 3, 
     delta=1, 
     delta0=0, 
     sd=3, 
     r = c(1,2,3), 
     r0 = c(1,2,3), p=NULL, p0=NULL, 
     alpha = 0.05, 
     power = 0.9, 
     ushape = "triangular", lshape = "triangular")

##########
# OUTPUT #
##########

# Design parameters for a 3 stage trial with 1 treatments:
#   
#   Stage 1 Stage 2 Stage 3
# Cumulative sample size per stage (control):      65     130     195
# Cumulative sample size per stage (active):       65     130     195
# 
# Maximum total sample size:  390 
# 
# Stage 1 Stage 2 Stage 3
# Upper bound:   2.132   1.885   1.847
# Lower bound:   0.000   1.131   1.847
# 
# 
# Simulated error rates based on 50000 simulations:
#   
#   Prop. rejecting at least 1 hypothesis:               0.901
# Prop. rejecting first hypothesis (Z_1>Z_2,...,Z_K)   0.901
# Prop. rejecting hypothesis 1:                        0.901
# Expected sample size:                              220.654
