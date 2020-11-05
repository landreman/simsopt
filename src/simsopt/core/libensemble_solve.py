# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides the least_squares_serial_solve
function. Eventually I can also put a serial_solve function here for
general optimization problems.
"""

import numpy as np
from scipy.optimize import least_squares
import logging
from mpi4py import MPI
from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, FINISHED_PERSISTENT_GEN_TAG
from libensemble.tools.gen_support import sendrecv_mgr_worker_msg
#from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens
from libensemble.tools.alloc_support import avail_worker_ids, sim_work, gen_work, count_persis_gens
from libensemble.libE import libE


"""
This file is based in part on libensemble/gen_funcs/persistent_uniform_sampling.py
"""

#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

def simsopt_libE_alloc(W, H, sim_specs, gen_specs, alloc_specs, persis_info):
    """
    This allocation function will give simulation work if possible, but
    otherwise start up to 1 persistent generator.  If all points requested by
    the persistent generator have been returned from the simulation evaluation,
    then this information is given back to the persistent generator.

    This function was adapted from libensemble/alloc_funcs/start_only_persistent.py
    """

    """
    W = list of workers
    """

    logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

    logger.debug('Allocator starting')
    
    Work = {}
    gen_count = count_persis_gens(W)

    #if persis_info.get('gen_started') and gen_count == 0:
    if 'gen_started' in persis_info and gen_count == 0:
        logger.debug('Exiting')
        # The one persistent worker is done. Exiting
        return Work, persis_info, 1

    # If i is in persistent mode, and all of its calculated values have
    # returned, give them back to i. Otherwise, give nothing to i
    for i in avail_worker_ids(W, persistent=True):
        gen_inds = (H['gen_worker'] == i)
        if np.all(H['returned'][gen_inds]):
            logger.debug('Giving values back to generator. Now H=\n{}\n{}'.format([i for i in H.dtype.fields], H))
            last_time_gen_gave_batch = np.max(H['gen_time'][gen_inds])
            inds_of_last_batch_from_gen = H['sim_id'][gen_inds][H['gen_time'][gen_inds] == last_time_gen_gave_batch]
            gen_work(Work, i,
                     sim_specs['in'] + [n[0] for n in sim_specs['out']] + [('sim_id')],
                     np.atleast_1d(inds_of_last_batch_from_gen), persis_info[i], persistent=True)

            #H['given_back'][inds_of_last_batch_from_gen] = True

    task_avail = ~H['given']
    for i in avail_worker_ids(W, persistent=False):
        if np.any(task_avail):
            # perform sim evaluations (if they exist in History).
            sim_ids_to_send = np.nonzero(task_avail)[0][0]  # oldest point
            logger.debug('Starting simulation. id={}'.format(sim_ids_to_send))
            sim_work(Work, i, sim_specs['in'], np.atleast_1d(sim_ids_to_send), persis_info[i])
            task_avail[sim_ids_to_send] = False

        elif gen_count == 0:
            logger.debug('Starting persistent generator')
            logger.debug('persis_info: {}'.format(persis_info))
            # Finally, call a persistent generator as there is nothing else to do.
            gen_count += 1
            gen_work(Work, i, gen_specs['in'], range(len(H)), persis_info[i],
                     persistent=True)
            persis_info['gen_started'] = True

    logger.debug('Allocator exiting')
    return Work, persis_info, 0


def simsopt_libE_sim(H, persis_info, sim_specs, libE_info):
    """
    This function is the "sim_f" function used by libEnsemble.
    """
    
    logger.debug('my_sim called with x={}'.format(H['x']))

    user_specs = sim_specs['user']    
    prob = user_specs['prob']

    # Create an array with a single element
    out = np.zeros(1, dtype=sim_specs['out'])

    x = H['x'][0]
    #out['y'][0, :] = prob.f(x)
    f = prob.f(x)
    #logger.info("f: {}  before out['y']: {}".format(f, out['y']))
    #out['y'] = prob.f(x)
    #out['y'] = f
    out['y'][0, :] = f
    #logger.info("f: {}  after out['y']: {}".format(f, out['y']))

    return out, persis_info


def residual_func(x, gen_specs, libE_info):
    logger.debug('residual_func called with x={}'.format(x))

    # Prepare batch of jobs to send to the manager, which in this case
    # has only a single job.
    out = np.zeros(1, dtype=gen_specs['out'])
    out['x'][0] = x

    # Send info to the manager:
    logger.debug('residual_func about to send work to manager.')
    tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], out)
    logger.debug("residual_func just returned from sendrecv. tag={} Work={}\ncalc_in={}".format(tag, Work, calc_in))
    #logger.debug("residual_func just returned from sendrecv. Work={}\ncalc_in={}\ncalc_in['y']={}".format(Work, calc_in, calc_in['y']))
    if tag in [STOP_TAG, PERSIS_STOP]:
        raise RuntimeError('Stop signal received')

    return calc_in['y'].flatten()


def jacobian_func(x, gen_specs, libE_info):
    logger.debug('Jacobian_func called with x={}'.format(x))

    user_specs = gen_specs['user']    
    eps = user_specs['eps']
    centered = user_specs['centered']
    
    # Prepare batch of jobs to send to the manager.
    # For now, assume 1-sided differences for simplicity
    nparams = len(x)
    out = np.zeros(nparams + 1, dtype=gen_specs['out'])
    out['x'][0] = x
    for j in range(nparams):
        out['x'][j + 1] = x
        out['x'][j + 1, j] = x[j] + eps

    # Send info to the manager:
    logger.debug('jacobian_func about to send work to manager.')
    tag, Work, calc_in = sendrecv_mgr_worker_msg(libE_info['comm'], out)
    logger.debug("jacobian_func just returned from sendrecv. Work={}\ncalc_in={}\ncalc_in['y']={}".format(Work, calc_in, calc_in['y']))
    if tag in [STOP_TAG, PERSIS_STOP]:
        raise RuntimeError('Stop signal received')

    nvals = len(calc_in['y'][0, :])
    jac = np.zeros((nvals, nparams))
    f0 = calc_in['y'][0, :]

    for j in range(nparams):
        f = calc_in['y'][j + 1, :]
        jac[:, j] = (f - f0) / eps
    
    logger.debug('jacobian_func returning with the following Jacobian:\n{}'.format(jac))
    return jac


def simsopt_libE_generator(H, persis_info, gen_specs, libE_info):
    """
    This function is the "gen_f" function used by libEnsemble.
    """
    logger.debug('Generator starting')

    user_specs = gen_specs['user']
    
    x0 = user_specs['x0']
    # For now, assume that we are using finite-difference gradients:
    try:
        result = least_squares(residual_func, x0, verbose=2, args=(gen_specs, libE_info), jac=jacobian_func)
    except RuntimeError as err:
        logger.info('Exiting scipy.optimize due to an exception: {}'.format(err))

    logger.debug('Generator exiting')
    persis_info['solution'] = result.x
    return None, persis_info, FINISHED_PERSISTENT_GEN_TAG


def least_squares_libE_solve(prob, grad=None, eps=1e-7, centered=False, **kwargs):
    """
    Solve a nonlinear-least-squares minimization problem using
    scipy.optimize and using libEnsemble for parallelized
    finite-difference gradients.

    prob should be a LeastSquaresProblem object.

    kwargs allows you to pass any arguments to scipy.optimize.least_squares.
    """
    logger.info("Beginning solve.")

    if MPI.COMM_WORLD.Get_size() < 2:
        raise RuntimeError("Must run with at least 3 MPI processes to use libEnsemble")
    
    prob._init() # In case 'fixed', 'mins', etc have changed since the problem was created.
    if grad is None:
        grad = prob.dofs.grad_avail
        
    #if not 'verbose' in kwargs:
        
    #libE_specs = {'comms': 'mpi', 'sim_dirs_make': True}
    libE_specs = {'comms': 'mpi'}

    gen_specs = {'gen_f': simsopt_libE_generator,
                 'out': [('x', float, (prob.dofs.nparams,))], # gen_f output (name, type, size)
                 'user': {
                     'grad': grad,
                     'eps': eps,
                     'centered': centered,
                     'x0': prob.x
                     }
                 }

    # We need to know the size of the output vector. This is only
    # known after one function evaluation. So if we don't know it yet,
    # do 1 function evaluation now:
    if prob.dofs.nvals is None:
        temp = prob.f()
    logger.info('prob.dofs.nvals: {}'.format(prob.dofs.nvals))
    sim_specs = {'sim_f': simsopt_libE_sim,
                 'in': ['x'], # Input field names. 'x' from gen_f output
                 'out': [('y', float, prob.dofs.nvals)], # output
                 'user': {
                     'prob': prob
                     }
                 }

    #alloc_specs = {'alloc_f': only_persistent_gens,
    alloc_specs = {'alloc_f': simsopt_libE_alloc,
                   'out': [('allocated', bool)]}

    # libE requires us to initialize persis_info with at least some keys to prevent errors.
    #persis_info = add_unique_random_streams({}, nworkers + 1) # Worker numbers start at 1
    persis_info = {}
    for j in range(MPI.COMM_WORLD.Get_size()):
        persis_info[j] = dict()

    exit_criteria = {'sim_max': 1000}

    # Main call to libEnsemble:
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs)

    proc0 = MPI.COMM_WORLD.Get_rank() == 0
    if proc0:
        print([i for i in H.dtype.fields])
        print(H)
        print("H['x']:", H['x'])
        print('libE_specs: ', libE_specs)

    logger.info("Completed solve. persis_info={}".format(persis_info))

    # Need to identify the optimum and broadcast it.
    if proc0:
        x = persis_info[1]['solution']
    else:
        x = np.copy(prob.x)
    # Make sure all procs get the optimal state vector:
    MPI.COMM_WORLD.Bcast(x)
    prob.dofs.set(x)
