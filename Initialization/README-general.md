While notebooks handle exercises, this is for initializing your work using HPC or any linux environment resource. 

# Basic Linux commands
One can find easily or ask a chatbot/search-engine to generate a list, I find this link useful: https://linux-commands.labex.io/

# Basic VI editor commands
If you are using Caltech's HPC, interactive mode (point your browser to https://interactive.hpc.caltech.edu
and login with your username password) allows you to edit files directly by opening in the left pan of an active session. However, this interactive mode depends on the availabity of compute nodes, and sometimes, a session may not be available. In those times, VI editor is strong tool to edit files, in an ssh logged-in session (that is, a command-terminal type login). 

This one is very easy to point to and learn some basic commands: https://github.com/pkrumins/bash-vi-editing-mode-cheat-sheet/blob/master/bash-vi-editing-mode-cheat-sheet.txt

# HPC Login
https://www.hpc.caltech.edu/documentation/faq/how-do-i-login-cluster

Also can be done, using https://interactive.hpc.caltech.edu , in this case, a terminal can be accessed by clicking "Terminal" option from the options available on the rightpane of a Jupyter lab session created. 

# Basic slurm job scheduler script
== Running a calculation without interrupting others. A minimal job submission scirpt is provided in this directory, "script.sh" that submits a DFT calculation (geometric optimization and frequency calculation) using ORCA Quantum Chemistry code input file "test.inp", which calls "water.xyz" - coordinates of a single molecule of water. 

The script is submitted using 

$ sbatch script.sh

More useful commands to monitor, cancel, inspect a slurm job are provided on - https://www.hpc.caltech.edu/documentation/slurm-commands

