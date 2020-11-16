# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides the run_standalone function for executing a
standalone code via subprocess.
"""

import subprocess
import time
import logging

logger = logging.getLogger(__name__)

def run_standalone(cmd: str,
                   code_name: str,
                   timeout: float = 1e100):
    """
    Run a standalone executable code.

    For now this is done using subprocess.run(). We may at some point
    want more sophisticated control over the subprocess, in which case
    we could move to subprocess.Popen().

    Args: 
      cmd: The command to be run, typically a list of strings
        corresponding to each argument.

    Returns: 
      True if the application completed with return code
      0. False if there was a different return code or if the timeout
      was reached.
    """

    stdout_file = open(code_name + '_stdout.txt', 'w')
    stderr_file = open(code_name + '_stderr.txt', 'w')

    logger.debug('Starting subprocess: {}'.format(cmd))
    timeout_bool = False
    try:
        completed_process = subprocess.run(cmd, stdout=stdout_file,
                                           stderr=stderr_file,
                                           timeout=timeout)
        returncode = completed_process.returncode
    except subprocess.TimeoutExpired:
        timeout_bool = True
        returncode = -1
        
    stdout_file.close()
    stderr_file.close()

    success = (not timeout_bool) and (returncode == 0)
    logger.debug('subprocess done. timeout:{} returncode:{} success:{}'\
                 .format(timeout_bool, returncode, success))
    return success
