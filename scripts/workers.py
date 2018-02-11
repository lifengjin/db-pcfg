#!/usr/bin/env python3

import logging
from multiprocessing import Process
import multiprocessing
import os, os.path
import signal
import subprocess
import sys
import time
import PyzmqWorker

def start_cluster_workers(work_distributer, cluster_cmd, maxLen, gpu):
    logging.debug("Cluster command is %s" % cluster_cmd)

    cmd_str = 'python3 %s/scripts/workers.py %s %d %d %d %d %d' % (os.getcwd(), work_distributer.host, work_distributer.jobs_port, work_distributer.results_port, work_distributer.models_port, maxLen, int(gpu))
    submit_cmd = [ cmd_arg.replace("%c", cmd_str) for cmd_arg in cluster_cmd.split()]
    logging.info("Making cluster submit call with the following command: %s" % str(submit_cmd))
    subprocess.call(submit_cmd)

def start_local_workers_with_distributer(work_distributer, maxLen, cpu_workers, gpu_workers=0, gpu=False, batch_size=1):
    logging.info("Starting workers with maxLen=%d and num_cpu_workers=%d and num_gpu_workers=%d" % (maxLen, cpu_workers, gpu_workers) )
    return start_local_workers(work_distributer.host, work_distributer.jobs_port, work_distributer.results_port, work_distributer.models_port, maxLen, cpu_workers, gpu_workers, gpu, batch_size)

def start_local_workers(host, jobs_port, results_port, models_port, maxLen, cpu_workers, gpu_workers=0, gpu=False, batch_size=1):
    logging.info("Starting %d cpu workers and %d gpu workers at host %s with jobs_port=%d, results_port=%d, models_port=%d, maxLen=%d" % (cpu_workers, gpu_workers, host, jobs_port, results_port, models_port, maxLen) )
    processes = []
    try:
        logging.info("Worker intializing GPU status: %s" % gpu)
        multiprocessing.set_start_method("spawn")
    except:
        logging.warning("context has been set!")
    for i in range(0, cpu_workers+gpu_workers):
        if i >= gpu_workers:
            gpu = False
            gpu_batch_size = 0 if gpu_workers > 0 else 1
        else:
            gpu = True
            gpu_batch_size = batch_size

        fs = PyzmqWorker.PyzmqWorker(host, jobs_port, results_port, models_port, maxLen, tid=i, gpu=gpu, batch_size=gpu_batch_size, level=logging.getLogger().getEffectiveLevel())
        signal.signal(signal.SIGTERM, fs.handle_sigterm)
        signal.signal(signal.SIGINT, fs.handle_sigint)
        signal.signal(signal.SIGALRM, fs.handle_sigalarm)
        p = Process(target=fs.run)
        processes.append(p)
        p.start()

    return processes

def main(args):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    if len(args) != 1 and len(args) != 7 and len(args) != 8:
        print("ERROR: Wrong number of arguments! Two run modes -- One argument of a file with properties or 7-8 arguments: <host (string)> <jobs port (int)> <results port (int)> <models port (int)> <max sentence length (int)> <gpu ([0|1])> <gpu batch size (int)> [num replicates (int)]")
        sys.exit(-1)

    if len(args) == 1:
        config_file = args[0] + "/masterConfig.txt"
        while True:
            if os.path.isfile(config_file):
                configs = open(config_file).readlines()
                if len(configs)==2 and 'OK' in configs[1]:
                    logging.info('OSC setup acquired. Starting a worker with ' + config_file)
                    args = configs[0].strip().split(' ')
                    break
            else:
                time.sleep(10)
    else:
        num_workers = 1
        if len(args) >= 8:
            num_workers = int(args[7])

    num_workers = 1
    if len(args) >= 8:
        num_workers = int(args[7])

    gpu = bool(int(args[5]))
    gpu_workers = int(gpu) * num_workers
    cpu_workers = (1 - int(gpu)) * num_workers

    processes = start_local_workers(host=args[0], jobs_port=int(args[1]), results_port=int(args[2]), models_port=int(args[3]), maxLen=int(args[4]), cpu_workers=cpu_workers, gpu_workers=gpu_workers, gpu=gpu, batch_size=int(args[6]))

    for i in range(0, num_workers):
        processes[i].join()

if __name__ == "__main__":
    main(sys.argv[1:])
