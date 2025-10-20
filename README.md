# Introductory work on PhD topics

This repository will serve as a central location for the introductory coding examples that I will use to explore efficient study designs and applying Bayesian optimization to characterize their statistical properties (e.g., sample sizes and error rates).

To run the RMarkdown file:
1. Ensure that you have Docker installed for your system.
2. Git clone the repository to your directory of choice.
3. `cd` to the directory in which you have cloned the repository.
4. Run `docker build -f phd-intro.Dockerfile -t phd-intro .` (ensure that the line ends in a `.`).  
     - Depending on your system, this command may take several minutes.
6. Run the command `docker run --rm --mount type=bind,source=.,dst=/project -ti -p 8787:8787 phd-intro`.
7. In your browser of choice, open `http://localhost:8787`.
8. Type in the username `rstudio` and the auto-generated password from the command line.

In order to see the files within the source directory:
1. Within the R console, `setwd("/project")`.
2. In the Files panel, under "More", click "Go To Working Directory".

Any files in the source directory should appear here. Any files created within this directory, will also be saved in the source directory.

Open the RMarkdown file, and run as normal.
