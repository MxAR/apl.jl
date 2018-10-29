# spawn as many workers as CPU threads are available
import Distributed.addprocs, Distributed.nworkers
while nworkers() < Sys.CPU_THREADS
	addprocs(1, enable_threaded_blas = true)
end

# enable BLAS and LAPACK to use those workers
import LinearAlgebra.BLAS.set_num_threads
set_num_threads(Sys.CPU_THREADS)
