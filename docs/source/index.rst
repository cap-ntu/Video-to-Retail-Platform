.. Hysia documentation master file, created by
   sphinx-quickstart on Fri Oct  5 10:37:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Hysia's documentation!
=================================

.. image:: https://travis-ci.com/HuaizhengZhang/Hysia.svg?token=SvqJmaGbqAbwcc7DNkD2&branch=master
    :target: https://travis-ci.com/HuaizhengZhang/Hysia

- In our system, we combine visual, audio and text information to understand the video content. To make full use of these models trained on multi-modality data, we design a multi-model schedule mechanism to fuse the features or predictions so the system can output more robust results.

- As Hysia schedules many deep learning models which are computationally expensive, we optimize our system jointly with each models both in training and inference stages.

- In order to be trade-off between computation resources (i.e., CPU and GPU) and the accuracy of prediction results, Hysia can adjust and generate the system configuration automatically so to process videos efficiently.


.. toctree::
   :maxdepth: 4
   :caption: Python API


   core
   models
   product
   tracking
   server
   optim
   exceptions
   install


**First: Check docs/build/html/index.html**


Install packages
==================

**You can use conda**

If you use conda, you can use all of functions that we supply.

`conda env create -f conda_environment.yml`

`source activate Hysia`

`conda remove --name Hysia --all`

**You can use pip**

But if you use pip, you can not use the search module

`pip install -r pip environment.yml`

How to use
==================

**Download pre-trained models**

`cd weights`

`python3 download_test_models.py`

**Set Django database**

`cd hysia/server/`

`sh set_db.sh`

**Use React to build the front-page**

`cd hysia/server/react-build`

`sh build.sh`


**Start model server**

`python model_server/start_model_servers.py`

`python manage.py runserver [server IP]:8000`

`browse [server IP]:8000`



.. toctree::
   :caption: C++ API Documentation

   cplus

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
