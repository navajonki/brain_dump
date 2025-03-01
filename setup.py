from setuptools import setup, find_packages

setup(
    name="zettelkasten",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.3.0",
        "openai-whisper @ git+https://github.com/openai/whisper.git",
        "pyyaml>=6.0.1",
        "tiktoken>=0.5.2",
        "chromadb>=0.4.22",
        "neo4j>=5.14.1",
        "python-dotenv>=1.0.0",
        "pydub>=0.25.1",
        "torch>=2.0.0",
        "soundfile>=0.12.1",
        "pytz>=2023.3",
        "tqdm>=4.65.0",
        "ffmpeg-python>=0.2.0",
        "deepgram-sdk>=2.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
        ],
    },
    python_requires=">=3.10",
    author="Zack Bodnar",
    author_email="zjbodnar@gmail.com",
    description="A tool for processing audio notes into a knowledge base",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
) 