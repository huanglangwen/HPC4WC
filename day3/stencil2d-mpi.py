# ******************************************************
#     Program: stencil2d
#      Author: Oliver Fuhrer
#       Email: oliverf@vulcan.com
#        Date: 20.05.2020
# Description: Simple stencil example
# ******************************************************

import time
import numpy as np
import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
from partitioner import Partitioner

comm = MPI.COMM_WORLD
num_ranks = comm.Get_size()
rank = comm.Get_rank()
buf = []


def laplacian( in_field, lap_field, num_halo, extend=0 ):
    """Compute Laplacian using 2nd-order centered differences.
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    num_halo  -- number of halo points
    
    Keyword arguments:
    extend    -- extend computation into halo-zone by this number of points
    """

    ib = num_halo - extend
    ie = - num_halo + extend
    jb = num_halo - extend
    je = - num_halo + extend
    
    lap_field[:, jb:je, ib:ie] = - 4. * in_field[:, jb:je, ib:ie]  \
        + in_field[:, jb:je, ib - 1:ie - 1] + in_field[:, jb:je, ib + 1:ie + 1 if ie != -1 else None]  \
        + in_field[:, jb - 1:je - 1, ib:ie] + in_field[:, jb + 1:je + 1 if je != -1 else None, ib:ie]


def update_halo( field, num_halo, p):
    """Update the halo-zone using an up/down and left/right strategy.
    
    field    -- input/output field (nz x ny x nx with halo in x- and y-direction)
    num_halo -- number of halo points
    
    Note: corners are updated in the left/right phase of the halo-update
    """
    global buf
    buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8 = buf
    # bottom edge (without corners)
    buf1[...] = field[:, -2 * num_halo:-num_halo, num_halo:-num_halo]
    buf3[...] = field[:, num_halo:2 * num_halo, num_halo:-num_halo]
    req1 = comm.Isend(buf1, dest = p.top())
    req2 = comm.Irecv(buf2,source = p.bottom())
    #field[:, 0:num_halo, num_halo:-num_halo] = field[:, -2 * num_halo:-num_halo, num_halo:-num_halo]
    
    # top edge (without corners)
    req3 = comm.Isend(buf3, dest = p.bottom())
    req4 = comm.Irecv(buf4, source = p.top())
    #field[:, -num_halo:, num_halo:-num_halo] = field[:, num_halo:2 * num_halo, num_halo:-num_halo]
    req1.Wait()
    req2.Wait()
    req3.Wait()
    req4.Wait()
    field[:, 0:num_halo, num_halo:-num_halo] = buf2[...]
    field[:, -num_halo:, num_halo:-num_halo] = buf4[...]

    buf5[...] = field[:, :, -2 * num_halo:-num_halo]
    buf7[...] = field[:, :, num_halo:2 * num_halo]
    # left edge (including corners)
    req5 = comm.Isend(buf5, dest = p.right())
    req6 = comm.Irecv(buf6, source = p.left())
    #field[:, :, 0:num_halo] = field[:, :, -2 * num_halo:-num_halo]
    
    # right edge (including corners)
    req7 = comm.Isend(buf7, dest = p.left())
    req8 = comm.Irecv(buf8, source = p.right())
    #field[:, :, -num_halo:] = field[:, :, num_halo:2 * num_halo]
    req5.Wait()
    req6.Wait()
    req7.Wait()
    req8.Wait()
    field[:, :, 0:num_halo] = buf6[...]
    field[:, :, -num_halo:] = buf8[...]
            

def apply_diffusion( in_field, out_field, alpha, num_halo, p, num_iter=1 ):
    """Integrate 4th-order diffusion equation by a certain number of iterations.
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    alpha     -- diffusion coefficient (dimensionless)
    
    Keyword arguments:
    num_iter  -- number of iterations to execute
    """

    tmp_field = np.empty_like( in_field )
    
    for n in range(num_iter):
        
        update_halo( in_field, num_halo, p)
        
        laplacian( in_field, tmp_field, num_halo=num_halo, extend=1 )
        laplacian( tmp_field, out_field, num_halo=num_halo, extend=0 )
        
        out_field[:, num_halo:-num_halo, num_halo:-num_halo] = \
            in_field[:, num_halo:-num_halo, num_halo:-num_halo] \
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]

        if n < num_iter - 1:
            in_field, out_field = out_field, in_field

            
@click.command()
@click.option('--nx', type=int, required=True, help='Number of gridpoints in x-direction')
@click.option('--ny', type=int, required=True, help='Number of gridpoints in y-direction')
@click.option('--nz', type=int, required=True, help='Number of gridpoints in z-direction')
@click.option('--num_iter', type=int, required=True, help='Number of iterations')
@click.option('--num_halo', type=int, default=2, help='Number of halo-pointers in x- and y-direction')
@click.option('--plot_result', type=bool, default=False, help='Make a plot of the result?')
def main(nx, ny, nz, num_iter, num_halo=2, plot_result=False):
    """Driver for apply_diffusion that sets up fields and does timings"""
    
    assert 0 < nx <= 1024*1024, 'You have to specify a reasonable value for nx'
    assert 0 < ny <= 1024*1024, 'You have to specify a reasonable value for ny'
    assert 0 < nz <= 1024, 'You have to specify a reasonable value for nz'
    assert 0 < num_iter <= 1024*1024, 'You have to specify a reasonable value for num_iter'
    assert 0 < num_halo <= 256, 'Your have to specify a reasonable number of halo points'
    
    p = Partitioner(comm, [nz, ny, nx], num_halo, periodic=(True, True))
    nyl, nxl = p.shape()[1], p.shape()[2]
    global buf
    buf = [np.zeros((nz,num_halo,nxl - 2 * num_halo)), np.zeros((nz,num_halo,nxl - 2 * num_halo)),
           np.zeros((nz,num_halo,nxl - 2 * num_halo)), np.zeros((nz,num_halo,nxl - 2 * num_halo)),
           np.zeros((nz,nyl,num_halo)), np.zeros((nz,nyl,num_halo)), 
           np.zeros((nz,nyl,num_halo)), np.zeros((nz,nyl,num_halo))]
    
    alpha = 1./32.
    in_field_global = np.zeros(1)
    if rank == 0:
        in_field_global = np.zeros( (nz, ny + 2 * num_halo, nx + 2 * num_halo) )
        in_field_global[nz // 4:3 * nz // 4, num_halo + ny // 4:num_halo + 3 * ny // 4, num_halo + nx // 4:num_halo + 3 * nx // 4] = 1.0
        np.save('in_field', in_field_global)
        if plot_result:
            plt.ioff()
            plt.imshow(in_field_global[in_field_global.shape[0] // 2, :, :], origin='lower')
            plt.colorbar()
            plt.savefig('in_field.png')
            plt.close()
    
    in_field = p.scatter(in_field_global, root = 0)
    out_field = np.copy( in_field )

    # warmup caches
    apply_diffusion( in_field, out_field, alpha, num_halo, p)

    # time the actual work
    tic = time.time()
    apply_diffusion( in_field, out_field, alpha, num_halo, p, num_iter=num_iter )
    toc = time.time()
    
    print("Elapsed time for work = {} s".format(toc - tic) )

    update_halo(out_field, num_halo, p)

    out_field_global = p.gather(out_field, root = 0)
    if rank == 0:
        np.save('out_field_mpi', out_field_global)
        if plot_result:
            plt.imshow(out_field_global[out_field_global.shape[0] // 2, :, :], origin='lower')
            plt.colorbar()
            plt.savefig('out_field_mpi.png')
            plt.close()


if __name__ == '__main__':
    main()
    


