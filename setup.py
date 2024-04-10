from setuptools import setup, find_packages

setup(
    name="transmaf",
    version="0.1.4",
    description="Probabilistic Time Series Modeling using Trans-MAF model.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Semen Eliseev",
    author_email="yeliseevsemyon@gmail.com",
    url="https://github.com/simonyelisey/trans-maf-forecasting",
    license="MIT",
    packages=find_packages(exclude=["tests", "reports", "time_grad", "trans_maf"]),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.8",
    install_requires=[
        "torch==2.1.0",
        "lightning==2.1.3",
        "pytorch-lightning==2.1.3",
        "protobuf~=3.20.3",
        "gluonts==0.14.3",
        "holidays",
        "pandas==2.2.0",
        "numpy==1.23.5",
        "matplotlib",
        "diffusers==0.25.1",
    ]
)
