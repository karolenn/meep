
Code for thesis work: 
Black-box optimization of simulated light extraction efficiency from quantum dots in pyramidal gallium nitride structures

Simulate EM fields for pyramidal GaN structures and calculates the light extraction efficiency (LEE). An optimization schema is introduced that
search a given variable space and attempts to find the pyramid design in the variable space with highest LEE.


TODO: \
-Round of values down to 1/resolution to remove ambiguity when meep rounds values\
-Process simulation results using e.g pandas. Using nested lists is pain when list depth is varying \
-Create coherent simstruct simulate function that allows input from arbitrary geometries and sources \
-Implement calculations for purcell effect \
-There is a possible bug that one result is written several time to json result causing rbf method to crash \
-and a lot more..

