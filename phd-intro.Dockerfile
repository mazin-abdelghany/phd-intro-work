# from the Rocker project, use rstudio Docker container
FROM rocker/rstudio:latest

# install posterior, tidybayes, ggdist, and loo
RUN install2.r --error \
     posterior \
     tidybayes \
     ggdist \
     loo \
<<<<<<< HEAD
     mvtnorm \
     gplite
=======
     mvtnorm
>>>>>>> d4bff225fe0483f68b96c04fd79d7713847650b7

# install RStan
RUN Rscript -e 'Sys.setenv(DOWNLOAD_STATIC_LIBV8 = 1)'
RUN Rscript -e 'install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)'

# run command, not detached
# docker run --rm --mount type=bind,src=.,dst=/project -ti -p 8787:8787 phd-intro

# run command, detached with password
# docker run -d -e PASSWORD=aeSoochif9moonga --rm --mount type=bind,src=.,dst=/project -ti -p 8787:8787 phd-intro
