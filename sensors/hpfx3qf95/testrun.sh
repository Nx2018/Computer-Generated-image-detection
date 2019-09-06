../../caffe_hpfx3/build/tools/caffe.bin test --model test.prototxt --weights snapshot_iter_80000.caffemodel --iterations 44282 --gpu 0 1> testresult-80k.txt 2>&1
../../caffe_hpfx3/build/tools/caffe.bin test --model test.prototxt --weights snapshot_iter_50000.caffemodel --iterations 44282 --gpu 0 1> testresult-50k.txt 2>&1

