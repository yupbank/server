#!/bin/bash
# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
export REPO_VERSION=$REPO_VERSION
export DATADIR=$DATADIR

export TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
export BACKEND_DIR=${TRITON_DIR}/backends
export TEST_JETSON=${TEST_JETSON:=0}
export CUDA_VISIBLE_DEVICES=0
export PYTHON_ENV_VERSION=${PYTHON_ENV_VERSION:="10"}

BASE_SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR} --log-verbose=1"
# Set the default byte size to 5MBs to avoid going out of shared memory. The
# environment that this job runs on has only 1GB of shared-memory available.
SERVER_ARGS="$BASE_SERVER_ARGS --backend-config=python,shm-default-byte-size=5242880"

PYTHON_BACKEND_BRANCH=$PYTHON_BACKEND_REPO_TAG
CLIENT_PY=./python_test.py
CLIENT_LOG="./client.log"
EXPECTED_NUM_TESTS="11"
TEST_RESULT_FILE='test_results.txt'
SERVER_LOG="./inference_server.log"
source ../common/util.sh
source ./common.sh

rm -fr *.log ./models

python3 --version | grep "3.10" > /dev/null
if [ $? -ne 0 ]; then
    echo -e "Expecting Python default version to be: Python 3.10 but actual version is $(python3 --version)"
    exit 1
fi

(bash -ex setup_python_enviroment.sh)

python3 --version | grep "3.${PYTHON_ENV_VERSION}" > /dev/null
if [ $? -ne 0 ]; then
    echo -e "Expecting Python version to be: Python 3.${PYTHON_ENV_VERSION} but actual version is $(python3 --version)"
    exit 1
fi

mkdir -p models/identity_fp32/1/
cp ../python_models/identity_fp32/model.py ./models/identity_fp32/1/model.py
cp ../python_models/identity_fp32/config.pbtxt ./models/identity_fp32/config.pbtxt
RET=0

cp -r ./models/identity_fp32 ./models/identity_uint8
(cd models/identity_uint8 && \
          sed -i "s/^name:.*/name: \"identity_uint8\"/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_UINT8/g" config.pbtxt && \
          sed -i "s/^max_batch_size:.*/max_batch_size: 8/" config.pbtxt && \
          echo "dynamic_batching { preferred_batch_size: [8], max_queue_delay_microseconds: 12000000 }" >> config.pbtxt)

cp -r ./models/identity_fp32 ./models/identity_uint8_nobatch
(cd models/identity_uint8_nobatch && \
          sed -i "s/^name:.*/name: \"identity_uint8_nobatch\"/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_UINT8/g" config.pbtxt && \
          sed -i "s/^max_batch_size:.*//" config.pbtxt >> config.pbtxt)

cp -r ./models/identity_fp32 ./models/identity_uint32
(cd models/identity_uint32 && \
          sed -i "s/^name:.*/name: \"identity_uint32\"/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_UINT32/g" config.pbtxt)

cp -r ./models/identity_fp32 ./models/identity_bool
(cd models/identity_bool && \
          sed -i "s/^name:.*/name: \"identity_bool\"/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_BOOL/g" config.pbtxt)

# Test models with `default_model_filename` variable set.
cp -r ./models/identity_fp32 ./models/default_model_name
mv ./models/default_model_name/1/model.py ./models/default_model_name/1/mymodel.py
(cd models/default_model_name && \
    sed -i "s/^name:.*/name: \"default_model_name\"/" config.pbtxt && \
    echo "default_model_filename: \"mymodel.py\"" >> config.pbtxt )

mkdir -p models/pytorch_fp32_fp32/1/
cp -r ../python_models/pytorch_fp32_fp32/model.py ./models/pytorch_fp32_fp32/1/
cp ../python_models/pytorch_fp32_fp32/config.pbtxt ./models/pytorch_fp32_fp32/
(cd models/pytorch_fp32_fp32 && \
          sed -i "s/^name:.*/name: \"pytorch_fp32_fp32\"/" config.pbtxt)

mkdir -p models/delayed_model/1/
cp -r ../python_models/delayed_model/model.py ./models/delayed_model/1/
cp ../python_models/delayed_model/config.pbtxt ./models/delayed_model/

mkdir -p models/init_args/1/
cp ../python_models/init_args/model.py ./models/init_args/1/
cp ../python_models/init_args/config.pbtxt ./models/init_args/

mkdir -p models/optional/1/
cp ../python_models/optional/model.py ./models/optional/1/
cp ../python_models/optional/config.pbtxt ./models/optional/

mkdir -p models/non_contiguous/1/
cp ../python_models/non_contiguous/model.py ./models/non_contiguous/1/
cp ../python_models/non_contiguous/config.pbtxt ./models/non_contiguous/config.pbtxt

# Unicode Characters
mkdir -p models/string/1/
cp ../python_models/string/model.py ./models/string/1/
cp ../python_models/string/config.pbtxt ./models/string

# More string tests
mkdir -p models/string_fixed/1/
cp ../python_models/string_fixed/model.py ./models/string_fixed/1/
cp ../python_models/string_fixed/config.pbtxt ./models/string_fixed

mkdir -p models/dlpack_identity/1/
cp ../python_models/dlpack_identity/model.py ./models/dlpack_identity/1/
cp ../python_models/dlpack_identity/config.pbtxt ./models/dlpack_identity

# Skip torch install on Jetson since it is already installed.
if [ "$TEST_JETSON" == "0" ]; then
  pip3 install torch==1.13.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
else
  # GPU tensor tests are disabled on jetson
  EXPECTED_NUM_TESTS=9
fi

# Disable env test for Jetson since cloud storage repos are not supported
# Disable ensemble, io and bls tests for Jetson since GPU Tensors are not supported
# Disable variants test for Jetson since already built without GPU Tensor support
# Disable decoupled test because it uses GPU tensors
if [ "$TEST_JETSON" == "0" ]; then
    SUBTESTS="bls decoupled"
    for TEST in ${SUBTESTS}; do
        # Run each subtest in a separate virtual environment to avoid conflicts
        # between dependencies.
        virtualenv --system-site-packages venv
        source venv/bin/activate

        (cd ${TEST} && bash -ex test.sh)
        if [ $? -ne 0 ]; then
        echo "Subtest ${TEST} FAILED"
        RET=1
        fi

        deactivate
        rm -fr venv
    done
fi

SUBTESTS="argument_validation custom_metrics request_rescheduling"
for TEST in ${SUBTESTS}; do
    # Run each subtest in a separate virtual environment to avoid conflicts
    # between dependencies.
    virtualenv --system-site-packages venv
    source venv/bin/activate

    (cd ${TEST} && bash -ex test.sh)

    if [ $? -ne 0 ]; then
        echo "Subtest ${TEST} FAILED"
        RET=1
    fi

    deactivate
    rm -fr venv
done

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
