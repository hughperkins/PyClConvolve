#!/bin/bash

~/envs/bin/python setup.py bdist_egg upload --identity=hughperkins --sign
~/env-34/bin/python setup.py bdist_egg upload --identity=hughperkins --sign
~/env-34/bin/python setup.py sdist upload --identity=hughperkins --sign


