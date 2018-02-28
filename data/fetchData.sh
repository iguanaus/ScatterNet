#!/bin/bash
# This file gathers all data. It can also be generated.

wget https://www.dropbox.com/sh/ii0bm0u4x8qsu9u/AABxSAPNlYn9HgeacX1LVv1ba?dl=1 -r -O data.zip
unzip data.zip
rm -rf data.zip
