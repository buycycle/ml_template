# Machine learning model cookiecutter template

# Description

Cookiecutter template for a machine learning project. Docker containerized app that is tested and deployed with Jenkins to an argoCD managed K8s cluster.

model/app.py
src/
tests/
notebooks/

are recommendation model specific which serves as an example for how to package the model in a flask app.


## Requirements
1. cookiecutter
2. ECR repo to push docker container to.
3. argoCD app

## Initialize

$cookiecutter git@gitlab.com:rauner/ml_project_cookiecutter_template.git

