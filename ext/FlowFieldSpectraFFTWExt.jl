module FlowFieldSpectraFFTWExt

using FFTW: FFTW
using FlowFieldSpectra: FlowFieldSpectra as FFS
using FlowTransformBindings: FlowTransformBindings as FTB

# FFTW implements AbstractFFTs for host arrays, so FFS's FFT plans (`FlowFieldSpectraAbstractFFTsExt`) run
# through it there, with FFTW's planner options: `MEASURE` times candidate algorithms on the planning
# array, for a plan held across executions, and `ESTIMATE` plans without running; `UNALIGNED` takes arrays
# whose start differs from the planning array's in FFTW's SIMD alignment. The thread count is the
# execution backend's.

FFS._available(::FFS._HostFFT) = true

FFS._fft_options(exec::FFS._HostExec, measure::Bool, aligned::Bool) =
    (; flags = (measure ? FFTW.MEASURE : FFTW.ESTIMATE) | (aligned ? zero(FFTW.UNALIGNED) : FFTW.UNALIGNED),
        num_threads = FFS._backend_nthreads(exec))

# NonuniformFFTs plans its host FFT through FFTW, with `MEASURE` unless its tag names flags.
FFS._planned_library(::FTB.NonuniformFFTsBackend{Nothing}, ::FFS._HostExec, measure::Bool) =
    FTB.NonuniformFFTsBackend(; fftw_flags = measure ? FFTW.MEASURE : FFTW.ESTIMATE)

end # module FlowFieldSpectraFFTWExt
