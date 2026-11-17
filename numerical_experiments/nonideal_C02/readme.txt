Validation of real gas riemann problems, based on:
"A Hybrid Real/Ideal Gas Mixture Computational Framework to
Capture Wave Propagation in Liquid Rocket Combustion
Chamber Conditions - D'Alessandro et al." described in the test-case of fig. 8. The time instant
of the comparison is not stated in the paper, so it assumed to be 0.00275 after some trial and error.

The jump in the rarefaction wave comes from the lacking entropy-fix in the Roe's solver. 
If the entropy fix is added (e.g. Harten-Hyman), the problem disappears.

2nd Order shows high-order reconstruction, using linear resontruction and Van Albada limiter.