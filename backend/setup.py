from setuptools import setup, find_packages

setup(
    name="rag_backend",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "flask",
        "chromadb",
        "openai",
        "sentence-transformers",
        "pypdf",
        "python-docx",
        "langchain",
        "tiktoken",
        "numpy",
    ],
    python_requires=">=3.8",
)
