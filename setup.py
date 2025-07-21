"""Setup configuration for otel_gen_ai_hydrator package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

with open("dev_requirements.txt", "r", encoding="utf-8") as fh:
    dev_requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="otel_gen_ai_hydrator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A toolkit for evaluating applications using distributed tracing data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/otel_gen_ai_hydrator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: System :: Monitoring",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "azure": [
            "azure-monitor-query>=1.2.0",
            "azure-identity>=1.15.0",
        ]
    },
    include_package_data=True,
    package_data={
        "otel_gen_ai_hydrator": ["templates/*.md", "examples/*.py"],
    },
)
