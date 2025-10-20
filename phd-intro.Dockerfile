# from the Rocker project, use rstudio Docker container
FROM rocker/rstudio:latest

# install posterior, tidybayes, ggdist, and loo
RUN install2.r --error \
     posterior \
     tidybayes \
     ggdist \
     loo 

# install RStan
RUN Rscript -e 'Sys.setenv(DOWNLOAD_STATIC_LIBV8 = 1)'
RUN Rscript -e 'install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)'

# run command 
# docker run --rm -ti -p 8787:8787 my/docker-tag
