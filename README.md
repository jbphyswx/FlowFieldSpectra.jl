# FlowFieldSpectra.jl
Easy Spectra of Flow Fields


This is menat to enable getting spectra of flow fields quickly and easily, on both structured and unstructured grids, in both cartesian and spherical coordinates.
This can be done with many Julia packages -- this one aims to make it simple by obscuring all the FFT setup and planning behind a unified interface.

It also provides utilities for calcluating energy spectra, for example going from a 2D flow field to a 1D energy spectrum by 2D FFT and radial integration.




We utilize FFTW (structured cartesian), FINUFFT (unstructured Cartesian), SphericalTransforms (structured spherical), and NUFSHT (unstructured spherical) for backends.
By default we only provide slow (N^2 complexity) naive, dependency-free direct methods. For your purposes, ensure to load your desired extension(s) to unlock fast NlogN methods.


Some other inspirational packages at https://github.com/FourierFlows/FourierFlows.jl (though this one adds entire equation solvers that are beyond the scope of this package)
