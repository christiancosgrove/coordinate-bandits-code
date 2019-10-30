mkdir data
cd data
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/aloi.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2
bzip2 -d *.bz2
