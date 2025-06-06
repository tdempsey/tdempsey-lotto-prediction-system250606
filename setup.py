from setuptools import setup, find_packages

setup(
    name="lotto_prediction_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
    ],
    extras_require={
        "web": ["flask>=2.0.0"],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A system for analyzing and predicting lottery numbers",
    keywords="lottery, prediction, data analysis",
    python_requires=">=3.6",
)