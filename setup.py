from setuptools import find_packages,setup
from typing import List


HYPEN_E_DOT="-e ."

def get_requirements(file_path:str) ->List:
    
    req=[]
    with open(file_path) as file:
        req=file.readlines()
        req= [r.replace("\n","")  for r in req]

        if HYPEN_E_DOT in req:
            req.remove(HYPEN_E_DOT)

    return req


setup(
    name="Bank Customer Chunk",
    version="0.0.1",
    author="soumya",
    author_email="soumya.ranjan.bhoi0011@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)