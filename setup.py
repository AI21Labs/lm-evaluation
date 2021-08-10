from setuptools import setup, find_packages

setup(name='lm-evaluation',
      version='0.1',
      description='LM Evaluation',
      url='https://github.com/AI21Labs/lm-evaluation',
      author='AI21 Labs',
      install_requires=[
            "tqdm",
            "requests",
            "pandas",
            "smart_open[gcs]"
      ],
      extras_require={},
      packages=find_packages(),
      zip_safe=True)
