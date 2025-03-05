import re
import sys
import os
from collections import defaultdict
from subprocess import Popen,PIPE

def train_NN_parallel(path_to_coords, path_to_glmpca, hidden_spatial, hidden_expression, 
                      output_dir, conda_environment, path_to_conda_folder,
                      epochs=10000, checkpoint=500, optimizer='adam', num_seeds=30, partition=None,
                     time="0-01:00:00"):

    hidden_spatial=' '.join(map(str, hidden_spatial))
    hidden_expression=' '.join(map(str, hidden_expression))
    
    tasks = 1
    mem_per_cpu = 5000
    processes = 1
    for seed in range(num_seeds):
        os.makedirs(f"{output_dir}/seed{seed}", exist_ok=True)
        cmd = f"gaston -i {path_to_coords} -o {path_to_glmpca} "
        cmd += f"--epochs {epochs} -d {output_dir} "
        cmd += f"--hidden_spatial {hidden_spatial} --hidden_expression {hidden_expression} "
        cmd += f"--optimizer {optimizer} --seed {seed} -c {checkpoint}"
        create_job_script(seed, f"{output_dir}/seed{seed}", tasks, processes, time, mem_per_cpu, cmd, conda_environment,path_to_conda_folder, partition=partition)

def create_job_script(name, outDir, tasks, cpuPerTask, time, mem_per_cpu, command, environment, path_to_conda_folder, partition=None):
    outFile = open('job_%s.sh' % name , 'w')
    o = outDir + "/out." + str(name)
    e = outDir + "/err." + str(name)
    print("#!/bin/bash", file=outFile)
    print("#SBATCH -J "+ str(name), file=outFile)
    print("#SBATCH -o " + o, file=outFile)
    print("#SBATCH -e " + e, file=outFile)
    print("#SBATCH --nodes=1", file=outFile)
    print("#SBATCH --ntasks=" + str(tasks), file=outFile)
    print("#SBATCH --cpus-per-task=" + str(cpuPerTask), file=outFile)
    print("#SBATCH -t " + str(time), file=outFile)
    print("#SBATCH --mem-per-cpu=" + str(mem_per_cpu), file=outFile)
    print(f"source {path_to_conda_folder} base", file=outFile)
    print(f"conda activate {environment}", file=outFile)
    print(command, file=outFile)
    outFile.close()
    jobId = sbatch_submit(outFile.name, partition)
    print(f'jobId: {jobId}')
    os.system("mv job_" + str(name) + ".sh " + outDir)
    return(jobId)

#Submit filename to slurm with sbatch, returns job id number
def sbatch_submit(filename, partition):
    print(f'partition: {partition}')
    if partition is not None:
        proc=Popen(f'sbatch -A {partition} {filename}',shell=True,stdout=PIPE,stderr=PIPE)
    else:
        proc = Popen('sbatch %s'%filename,shell=True,stdout=PIPE,stderr=PIPE)
    stdout,stderr = proc.communicate()
    stdout = stdout.decode("utf-8","ignore")
    stdout = stdout.strip()
    stdout = stdout.strip('Submitted batch job ')
    return(stdout)
