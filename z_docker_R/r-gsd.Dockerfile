# from the Rocker project, use rstudio Docker container
FROM rocker/rstudio:latest

# install posterior, tidybayes, ggdist, and loo
RUN install2.r --error \
     gplite \
     mvtnorm \
     plotly \
     bench \
     rpact \
     profvis \
     tictoc \
     gridExtra 

RUN Rscript -e 'install.packages("devtools", dependencies = TRUE)'

WORKDIR /project

RUN chown -R rstudio:rstudio /project
RUN chmod -R a+rwX /project

## install RStan
# RUN Rscript -e 'Sys.setenv(DOWNLOAD_STATIC_LIBV8 = 1)'
# RUN Rscript -e 'install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)'

# run command, not detached
# docker run --rm --mount type=bind,src=.,dst=/project -ti -p 8787:8787 rstudio-gsd

# run command, detached with password
# docker run -d -e PASSWORD=aeSoochif9moonga --rm --mount type=bind,src=.,dst=/project -ti -p 8787:8787 rstudio-gsd
