from setuptools import find_packages, setup

setup(
    name="autovideo",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    version=0.1,
    description="Everything to build an automated video",
    author="Me",
)
