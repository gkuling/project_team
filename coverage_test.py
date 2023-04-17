
with open("/home/gkuling/project_team/MNIST_classification_TrainTestSplit.py") as f:
    exec(f.read())

with open(
        "/home/gkuling/project_team/MNIST_classification_KFoldValidation.py") \
        as f:
    exec(f.read())


with open("/home/gkuling/project_team/MNIST_classification_HPTuning.py") as f:
    exec(f.read())