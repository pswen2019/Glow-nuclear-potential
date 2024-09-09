# Generate chiral potential samples
Generate n3lo chiral potential samples whose LECs follows normal distribution
at $\Lambda = 450, 500, 550$MeV.

```bash
python ./multiprocess_pot_generate_normal_distribution_for_LEC.py --numprocess 10 --numpot 30
```
The above cmd generate n3lo chiral potentials using 10 process. 
Each process generates 30 chiral potential samples.
The potential samples are generated in `allpot` directory. 
The LECs and $\Lambda$ corresponding to each samples are recorded in `allpot/label.txt`
