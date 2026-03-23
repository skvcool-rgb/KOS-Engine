from setuptools import setup, find_packages

setup(
    name="kos-engine",
    version="4.1.0",
    description="KOS: A Fuel-Constrained Spreading Activation Engine for Zero-Hallucination Knowledge Retrieval",
    author="Suraj",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.0.0",
        "nltk>=3.8.0",
        "jellyfish>=1.0.0",
        "sympy>=1.12",
        "sentence-transformers>=2.2.0",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "networkx>=3.0",
        "pyvis>=0.3.0",
    ],
    extras_require={
        "ui": ["streamlit>=1.30.0"],
        "prover": ["z3-solver>=4.12.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
)
