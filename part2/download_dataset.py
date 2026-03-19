from roboflow import Roboflow
rf = Roboflow(api_key="zcApQr54bnD6POqEKFmP")
project = rf.workspace("maya-mlots").project("dog-nmpmi-f1sao")
version = project.version(2)
dataset = version.download("voc")
