import numpy as np 
import os 
import subprocess
import shutil
from multiprocessing import Process
import random
from tqdm import tqdm

rseed = random.randint(1, 1000000)

cwd = os.getcwd()
if os.path.exists("pot"):
    shutil.rmtree("pot")
if os.path.exists("allpot"):
    shutil.rmtree("allpot")

if not os.path.exists("pot"):
    os.makedirs("pot")

lecn2lo450 = np.loadtxt("Chiral2017LEC_N2LO450.dat", unpack = True)
lecn2lo500 = np.loadtxt("Chiral2017LEC_N2LO500.dat", unpack = True)
lecn2lo550 = np.loadtxt("Chiral2017LEC_N2LO550.dat", unpack = True)
lecn2 = [lecn2lo450, lecn2lo500, lecn2lo550]
diff450to500 = np.abs(lecn2lo450 - lecn2lo500)
diff450to550 = np.abs(lecn2lo450 - lecn2lo550)
diff500to550 = np.abs(lecn2lo500 - lecn2lo550)

lecmax = np.max([lecn2lo450, lecn2lo500, lecn2lo550], axis = 0)
lecmin = np.min([lecn2lo450, lecn2lo500, lecn2lo550], axis = 0)

delta = lecmax - lecmin
print(delta)
width = delta / 8

def generatepot(processnum, numpot):
    np.random.seed(processnum + rseed)
    for i in tqdm(range(numpot), ncols = 54):

        potid = i % 3
        lamb = int(lecn2[potid][-1])

        dirname = os.path.join("./pot", "{:04d}".format(processnum))
        datadir = os.path.join(dirname, "pot")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        shutil.copy("a.out", dirname)
        os.chdir(dirname)

        leclist = np.random.normal(loc = lecn2[potid], scale = width)
        leclist[1] = leclist[3]
        leclist[-1] = lecn2[potid][-1]
        np.savetxt("./lec.dat", leclist, fmt = "%05.6f")
        cmd = "./a.out"
        subprocess.run(cmd)

        potfilename = "pot/{:d}{:03d}_{:03d}{:06d}.dat".format(
                2, lamb, processnum, i)
        shutil.copy('pot/pot.dat', potfilename)
        os.remove("lec.dat")
        os.remove("pot/pot.dat")

        lecfile = 'pot/label.txt'
        if not os.path.isfile(lecfile):
            f = open(lecfile, 'w')
            f.close()
        with open(lecfile, "ab") as f:
            label = np.array([
                "{:d}{:03d}_{:03d}{:06d}.dat".format(
                    2, lamb, processnum, i)])
            templabellist = [2, lamb]
            labellist =["{:03d}".format(i) for i in templabellist]
            label = np.append(label, labellist)
            np.savetxt(f, np.c_[[label]], fmt = "%s", delimiter = " ")

        os.chdir(cwd)


if __name__ == "__main__":

    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--numprocess', type = int, default = 10, 
                        help = 'the number of process generating potential samples')
    parser.add_argument('--numpot', type = int, default = None, 
                        help = 'the number potential samples generated from each process')
    args = parser.parse_args()
    config = vars(args)

    numprocess = config['numprocess']
    numpot = config['numpot'] if config['numpot'] is not None else 3*numprocess
    
    if os.path.exists('a.out'):
        os.remove('a.out')
    subprocess.call(['ifort', 'Chiral2017_N2LO_PotGen.f', '-O3'])
    
    processelist = [
            Process(
                target = generatepot, 
                args = (i,numpot)
                ) for i in range(numprocess)]
    for iprocess in processelist:
        iprocess.start()
    for iprocess in processelist:
        iprocess.join()


    rootdir = "allpot"
    rootlabelfile = os.path.join(rootdir, "alllabel.txt")
    if os.path.exists(rootdir):
        shutil.rmtree(rootdir)
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)
    f = open(rootlabelfile, "w")
    f.close()

    cwd = os.getcwd()

    with open(rootlabelfile, "w") as alllabelfile:
        for process in range(numprocess):
            labelfilename = os.path.join(
                    cwd, "pot", "{:04d}".format(process), 
                    "pot", "label.txt")
            with open(labelfilename, "r") as labelfile:
                for line in labelfile:
                    alllabelfile.write(line)

    for process in range(numprocess):
        processdir = os.path.join(
                "pot", "{:04d}".format(process), "pot")
        shutil.copytree(processdir, rootdir, dirs_exist_ok=True)

    momentafile = os.path.join(
            cwd, "pot", "{:04d}".format(process), "Momenta.dat"
            )
    shutil.copy(momentafile, rootdir)
    shutil.rmtree("pot")
    os.remove("allpot/label.txt")
    os.remove('./a.out')
