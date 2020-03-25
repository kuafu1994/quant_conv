# CMake generated Testfile for 
# Source directory: /home/pfzhang/workspace/quant_conv
# Build directory: /home/pfzhang/workspace/quant_conv/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test-conv "test-conv")
subdirs("googletest")
subdirs("googlebenchmark")
