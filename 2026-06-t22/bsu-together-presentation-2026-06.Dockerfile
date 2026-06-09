# from the Rocker project, use rstudio Docker container
FROM rocker/rstudio:latest

RUN install2.r --error \
     gplite \
     mvtnorm \
     plotly \
     gridExtra \
     plgp

# run command, not detached
# docker run --rm --mount type=bind,src=.,dst=/project -ti -p 8787:8787 esd-presentation

# run command, detached with password
# docker run -d -e PASSWORD=aeSoochif9moonga --rm --mount type=bind,src=.,dst=/project -ti -p 8787:8787 esd-presentation
dock