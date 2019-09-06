cd /home/forensics/CGPG_experiment/sensor/imagepatches/patchPG2000qf95/

find ./test -name *.bmp | grep Real | sort | cut -d '/' -f3-4 | sed 's/$/ 1/g' >> test.txt
find ./test -name *.bmp | grep CGG | sort | cut -d '/' -f3-4 | sed 's/$/ 0/g' >> test.txt

find ./train -name *.bmp | grep Real | sort | cut -d '/' -f3-4 | sed 's/$/ 1/g' >> train.txt
find ./train -name *.bmp | grep CGG | sort | cut -d '/' -f3-4 | sed 's/$/ 0/g' >> train.txt

find ./validation -name *.bmp | grep Real | sort | cut -d '/' -f3-4 | sed 's/$/ 1/g' >> val.txt
find ./validation -name *.bmp | grep CGG | sort | cut -d '/' -f3-4 | sed 's/$/ 0/g' >> val.txt


cd /home/forensics/CGPG_experiment/sensor/imagepatches/patchPG2000qf85/

find ./test -name *.bmp | grep Real | sort | cut -d '/' -f3-4 | sed 's/$/ 1/g' >> test.txt
find ./test -name *.bmp | grep CGG | sort | cut -d '/' -f3-4 | sed 's/$/ 0/g' >> test.txt

find ./train -name *.bmp | grep Real | sort | cut -d '/' -f3-4 | sed 's/$/ 1/g' >> train.txt
find ./train -name *.bmp | grep CGG | sort | cut -d '/' -f3-4 | sed 's/$/ 0/g' >> train.txt

find ./validation -name *.bmp | grep Real | sort | cut -d '/' -f3-4 | sed 's/$/ 1/g' >> val.txt
find ./validation -name *.bmp | grep CGG | sort | cut -d '/' -f3-4 | sed 's/$/ 0/g' >> val.txt


cd /home/forensics/CGPG_experiment/sensor/imagepatches/patchPG2000qf75/

find ./test -name *.bmp | grep Real | sort | cut -d '/' -f3-4 | sed 's/$/ 1/g' >> test.txt
find ./test -name *.bmp | grep CGG | sort | cut -d '/' -f3-4 | sed 's/$/ 0/g' >> test.txt

find ./train -name *.bmp | grep Real | sort | cut -d '/' -f3-4 | sed 's/$/ 1/g' >> train.txt
find ./train -name *.bmp | grep CGG | sort | cut -d '/' -f3-4 | sed 's/$/ 0/g' >> train.txt

find ./validation -name *.bmp | grep Real | sort | cut -d '/' -f3-4 | sed 's/$/ 1/g' >> val.txt
find ./validation -name *.bmp | grep CGG | sort | cut -d '/' -f3-4 | sed 's/$/ 0/g' >> val.txt
