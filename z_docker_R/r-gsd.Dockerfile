# from the Rocker project, use rstudio Docker container
FROM rocker/rstudio:latest

# install packages of interest 
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

## to install RStan, uncomment the below
# RUN Rscript -e 'Sys.setenv(DOWNLOAD_STATIC_LIBV8 = 1)'
# RUN Rscript -e 'install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)'

WORKDIR /project

RUN chown -R rstudio:rstudio /project
RUN chmod -R a+rwX /project