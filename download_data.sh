# download data for test
apt-get update -y
apt-get install -y gnumeric
mkdir data
cd data && curl -O 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
ls -a
ssconvert default%20of%20credit%20card%20clients.xls data1.csv 
echo 'this is ls -a'
ls -a
grep -v "X1" data1.csv > data.csv
echo 'this is ls -a after grep'
