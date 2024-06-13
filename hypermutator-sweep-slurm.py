#!/opt/software/Python/3.6.4-foss-2018a/bin/python3
#SBATCH --time=1:00:00
#SBATCH --job-name mutxplore
#SBATCH --output="/mnt/home/%u/joblog/id=%A_%a+ext.txt"
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH --array=0-799
#SBATCH --requeue

# %%
import datetime
import itertools as it
import os
import shutil
import sys
import tempfile
import uuid

import numpy as np
import pandas as pd
import tqdm as tq
from tqdm import tqdm

print("date", datetime.datetime.now())
print("sys.version", sys.version)
print("numpy", np.__version__)
print("pandas", pd.__version__)
print("tqdm", tq.__version__)


# %%
HOME = os.environ.get("HOME")
SLURM_ARRAY_JOB_ID = os.environ.get("SLURM_ARRAY_JOB_ID", "nojid")
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "nojid")
SLURM_ARRAY_TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

print("HOME={} SLURM_ARRAY_JOB_ID={}".format(HOME, SLURM_ARRAY_JOB_ID))
print(
    "SLURM_JOB_ID={} SLURM_ARRAY_TASK_ID={}".format(
        SLURM_JOB_ID, SLURM_ARRAY_TASK_ID
    )
)

# %%
params = list(
    it.product(
        range(1000),
        np.linspace(2, 6.75, 20),
    ),
)

REPLICATE, LOG10_NPOP = params[SLURM_ARRAY_TASK_ID]
print("REPLICATE={} LOG10_NPOP={}".format(REPLICATE, LOG10_NPOP))

# %%
LOG2_MUT_P = -8
LOG2_BENEFICIAL_P = LOG2_MUT_P + -14
LOG2_DELETERIOUS_P = LOG2_MUT_P + 0
LOG2_MUTATOR_P = LOG2_MUT_P + -2

LOG10_NPOP = 2
NPOP = int(10**LOG10_NPOP)

NGEN_MAX = 1000000
NGEN_EVERY = 1000

IDX_MUTATOR = 0
IDX_BENEFICIAL = 1
IDX_DELETERIOUS = 2

SELECTION_COEFFICIENT = 0.1

REPLICATE = 0

# %%
np.random.seed(SLURM_ARRAY_TASK_ID)


# %%
def initpop():
    return np.zeros((NPOP, 3), dtype=np.int64)


# %%
def mutpop(pop):  # : np.array
    m = pop[:, IDX_MUTATOR].astype(np.float64)
    pop[:, IDX_MUTATOR] += np.random.poisson(
        2 ** (LOG2_MUTATOR_P),
        NPOP,
    )  #  + m
    pop[:, IDX_BENEFICIAL] += np.random.poisson(
        2 ** (LOG2_BENEFICIAL_P + m),
        NPOP,
    )
    pop[:, IDX_DELETERIOUS] += np.random.poisson(
        2 ** (LOG2_DELETERIOUS_P + m),
        NPOP,
    )
    return pop


# %%
def selectpop(pop):  # : np.array
    NSELECTED = int(NPOP * SELECTION_COEFFICIENT)
    NRANDOMIZED = NPOP - NSELECTED

    # Generate random indices for the tournament selection
    idx1 = np.random.randint(NPOP, size=NSELECTED)
    idx2 = np.random.randint(NPOP, size=NSELECTED)

    # Calculate the benefit-harm values for both indices
    benefit_harm1 = pop[idx1, IDX_BENEFICIAL] - pop[idx1, IDX_DELETERIOUS]
    benefit_harm2 = pop[idx2, IDX_BENEFICIAL] - pop[idx2, IDX_DELETERIOUS]

    # Select the population based on benefit-harm comparison
    condition = benefit_harm1 > benefit_harm2
    selected_pop = np.where(condition[:, np.newaxis], pop[idx1], pop[idx2])

    # Create the new population array
    new_pop = np.empty_like(pop)
    new_pop[:NSELECTED] = selected_pop

    # Introduce random selection for the rest of the population
    random_indices = np.random.choice(NPOP, NRANDOMIZED, replace=True)
    new_pop[NSELECTED:] = pop[random_indices]

    return new_pop


# %%
def evolvepop(yield_every=1000):  # : int

    uu = uuid.uuid4()
    pop = initpop()
    for gen in range(NGEN_MAX):
        pop = mutpop(pop)
        pop = selectpop(pop)

        if gen % yield_every == 0:
            yield {
                "REPLICATE": REPLICATE,
                "NPOP": NPOP,
                "NGEN": NGEN_MAX,
                "generation": gen,
                "LOG2_MUT_P": LOG2_MUT_P,
                "LOG2_BENEFICIAL_P": LOG2_BENEFICIAL_P,
                "LOG2_DELETERIOUS_P": LOG2_DELETERIOUS_P,
                "LOG2_MUTATOR_P": LOG2_MUTATOR_P,
                "SELECTION_COEFFICIENT": SELECTION_COEFFICIENT,
                "mean beneficial": pop[:, IDX_BENEFICIAL].mean(),
                "median beneficial": np.median(pop[:, IDX_BENEFICIAL]),
                "min beneficial": np.min(pop[:, IDX_BENEFICIAL]),
                "max beneficial": np.max(pop[:, IDX_BENEFICIAL]),
                "mean deleterious": pop[:, IDX_DELETERIOUS].mean(),
                "median deleterious": np.median(pop[:, IDX_DELETERIOUS]),
                "min deleterious": np.min(pop[:, IDX_DELETERIOUS]),
                "max deleterious": np.max(pop[:, IDX_DELETERIOUS]),
                "mean mutator": pop[:, IDX_MUTATOR].mean(),
                "median mutator": np.median(pop[:, IDX_MUTATOR]),
                "min mutator": np.min(pop[:, IDX_MUTATOR]),
                "max mutator": np.max(pop[:, IDX_MUTATOR]),
                "uuid": uu,
            }


# %%

if __name__ == "__main__":
    outpath = "{}/SLURM_ARRAY_JOB_ID={}".format(HOME, SLURM_ARRAY_JOB_ID)
    print("outpath {}".format(outpath))
    os.makedirs(outpath, exist_ok=True)

    outname = "{}/SLURM_ARRAY_TASK_ID={}+ext=.csv".format(
        outpath, SLURM_ARRAY_TASK_ID
    )
    print("outname {}".format(outname))

    records = []
    for record in tqdm(evolvepop(), total=NGEN_MAX // NGEN_EVERY):
        records.append(record)
        df = pd.DataFrame.from_records(records)
        tempname = tempfile.NamedTemporaryFile(delete=False).name
        df.to_csv(tempname, index=False)
        shutil.move(tempname, outname)
        shutil.rmtree(tempname, ignore_errors=True)
